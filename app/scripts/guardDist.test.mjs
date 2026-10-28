// @vitest-environment node
/**
 * Tests for the deploy guard (scripts/guardDist.mjs, ADR 0001 D2).
 *
 * Each test builds a small app directory in a temp folder (dist/, public/, src/ and whatever else the
 * case needs) and runs the real check on it. A fixture is data: nothing here re-implements the
 * guard. Every expectation is a count or a path that can be read off the fixture.
 */
import { spawnSync } from 'node:child_process'
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { checkBuild, main } from './guardDist.mjs'

const SCRIPT = join(dirname(fileURLToPath(import.meta.url)), 'guardDist.mjs')
const SYNTHETIC = '{"data_mode":"synthetic","dataset":"x"}'
const REAL = '{"data_mode":"real","dataset":"x"}'

let root // the app directory under test
let outsideRoot // a folder OUTSIDE the app that holds real data: what a symlink inside the app would point at
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'guard-dist-'))
  outsideRoot = mkdtempSync(join(tmpdir(), 'guard-dist-outside-'))
  writeFileSync(join(outsideRoot, 'real.json'), '{"data_mode":"real"}')
})
afterEach(() => {
  rmSync(root, { recursive: true, force: true })
  rmSync(outsideRoot, { recursive: true, force: true })
})

/** Write `content` at root/<rel>, creating folders. */
function put(rel, content = '') {
  const abs = join(root, rel)
  mkdirSync(dirname(abs), { recursive: true })
  writeFileSync(abs, content)
}

/** A build that is safe to ship: one page, one script, synthetic JSON in dist/, public/ and src/. */
function cleanApp() {
  put('dist/index.html', '<!doctype html><title>app</title>')
  put('dist/assets/index-abc.js', 'console.log("hello")')
  put('dist/data/synthetic/manifest.json', SYNTHETIC)
  put('public/data/synthetic/manifest.json', SYNTHETIC)
  put('src/data/fixture.json', '{"data_mode":"synthetic","schema_version":1,"v":[1,2]}')
}

/** "rule:path" strings, sorted, so a test names every violation and ignores the order they were found in. */
const rules = (result) => result.violations.map((v) => `${v.rule}:${v.path}`).sort()

describe('a clean build', () => {
  it('has no violations and reports how many files it read', () => {
    cleanApp()
    const result = checkBuild(root)
    expect(result.violations).toEqual([])
    // dist: index.html + assets/index-abc.js + data/synthetic/manifest.json = 3 files read for the marker;
    // elsewhere: public/data/synthetic/manifest.json + src/data/fixture.json = 2;
    // JSON checked against the allowlist: one in dist, one in public, one in src = 3.
    expect(result.scanned).toEqual({ dist: 3, app: 2, json: 3 })
  })
  it('does not mind a missing public/ or src/ folder (a build input can be absent)', () => {
    put('dist/index.html', '<html></html>')
    expect(checkBuild(root)).toEqual({ violations: [], scanned: { dist: 1, app: 0, json: 0 } })
  })
  it('ignores non-data files: a favicon, a stylesheet, a README, other JSON-free assets', () => {
    cleanApp()
    put('public/favicon.svg', '<svg/>')
    put('src/index.css', 'body{}')
    put('README.md', 'prose')
    put('index.html', '<html/>')
    expect(checkBuild(root).violations).toEqual([])
  })
})

describe('the real-data marker in dist/', () => {
  // Every spelling below must be caught. Spellings 4 and 5 are the ones a bundle can contain:
  // an object literal with an unquoted key, and a JSON string embedded in JS with escaped quotes.
  const SPELLINGS = [
    ['compact JSON', '{"data_mode":"real"}'],
    ['spaces around the colon', '{"data_mode" : "real"}'],
    ['pretty-printed with a newline after the colon', '{\n  "data_mode":\n    "real"\n}'],
    ['unquoted key in a JS object literal', 'export default {data_mode:"real",v:[1,2]}'],
    ['JSON inside a JS string, quotes escaped', 'JSON.parse("{\\"data_mode\\":\\"real\\",\\"v\\":[1]}")'],
    ['single quotes', "x={'data_mode':'real'}"],
    ['the marker is not first in the file', `${'x'.repeat(5000)}{"a":1,"data_mode":"real"}`],
  ]
  it.each(SPELLINGS)('is caught: %s', (_name, body) => {
    cleanApp()
    put('dist/assets/leak.js', body)
    expect(rules(checkBuild(root))).toEqual(['real-marker:dist/assets/leak.js'])
  })

  // None of these is the marker, so none may fail a build.
  const NOT_THE_MARKER = [
    ['synthetic', '{"data_mode":"synthetic"}'],
    ['a longer word starting with real', '{"data_mode":"realistic"}'],
    ['a longer word ending in real', '{"data_mode":"unreal"}'],
    ['real under a different key', '{"mode":"real"}'],
    ['the key alone', '{"data_mode":null}'],
    ['two words far apart', '{"data_mode":"synthetic","note":"not real"}'],
    ['a comparison, not a value', "if (doc.data_mode === 'real') {}"],
  ]
  it.each(NOT_THE_MARKER)('is not flagged: %s', (_name, body) => {
    cleanApp()
    put('dist/assets/ok.js', body)
    expect(checkBuild(root).violations).toEqual([])
  })

  it('finds the marker in a nested folder and in a non-text file, and names each path once', () => {
    cleanApp()
    put('dist/a/b/c/deep.js', '{"data_mode":"real"}')
    // 4 arbitrary bytes, the marker, 3 more bytes: a binary file still counts.
    writeFileSync(join(root, 'dist/logo.png'), Buffer.concat([Buffer.from([0x89, 0x50, 0xff, 0x00]), Buffer.from('{"data_mode":"real"}'), Buffer.from([0xfe, 0xff, 0x00])]))
    expect(rules(checkBuild(root))).toEqual(['real-marker:dist/a/b/c/deep.js', 'real-marker:dist/logo.png'])
  })

  it('never prints the contents of the file it flags, nor a JSON parser message that quotes them', () => {
    cleanApp()
    // Not valid JSON (the bare word), so a parse error would quote "SECRET-VALUE-4242" if it were printed.
    put('dist/data/leak.json', '{"data_mode":"real","secret_close":SECRET-VALUE-4242}')
    const errors = []
    const code = main(['--root', root], { log: () => {}, error: (m) => errors.push(m) })
    expect(code).toBe(1)
    const printed = errors.join('\n')
    expect(printed).toContain('dist/data/leak.json')
    // V8's JSON.parse message quotes about ten characters either side of the fault, so a partial
    // quote ("secret_close", "SECRET-VAL") is what a leak would look like, not the whole value.
    expect(printed).not.toContain('SECRET')
    expect(printed).not.toContain('secret_close')
  })
})

describe('the marker anywhere in the app tree, not only in src/ and public/ (a bundler hides it from the dist/ scan)', () => {
  // Real JSON copied outside src/ and public/ and imported by a relative path or import.meta.glob.
  const PLACES = [
    'realdata/leak.json',
    'data/term_structure.json',
    'fixtures/real.json',
    '.local/leak.json',
    'src/lib/loader.ts',
    'src/data/leak.ts',
    'scripts/other.mjs',
    'index.html',
    '.env.local',
    'src/node_modules/leak.json',
  ]
  it.each(PLACES)('is caught in %s', (rel) => {
    cleanApp()
    put(rel, REAL)
    expect(rules(checkBuild(root))).toContain(`real-marker:${rel}`)
  })

  it('names exactly the one planted file for a JSON outside src/ and public/', () => {
    cleanApp()
    put('realdata/leak.json', REAL)
    expect(rules(checkBuild(root))).toEqual(['real-marker:realdata/leak.json'])
  })

  it('does not look inside node_modules/, coverage/ or .git/ at the top level', () => {
    cleanApp()
    put('node_modules/pkg/data.json', REAL)
    put('coverage/lcov-report/x.html', REAL)
    put('.git/config', REAL)
    expect(checkBuild(root).violations).toEqual([])
  })

  it('does not skip a node_modules/ that is nested: Vite resolves relative imports into it', () => {
    cleanApp()
    put('src/data/node_modules/x.json', REAL)
    expect(rules(checkBuild(root))).toContain('real-marker:src/data/node_modules/x.json')
  })

  it('exempts only files that describe the marker: docs, tests and the guard itself', () => {
    cleanApp()
    const prose = 'The marker looks like {"data_mode":"real"}.'
    for (const rel of ['README.md', 'docs/NOTES.MD', 'src/lib/loader.test.ts', 'src/App.test.tsx', 'scripts/other.test.mjs', 'scripts/guardDist.mjs']) put(rel, prose)
    expect(checkBuild(root).violations).toEqual([])
  })

  it('does not exempt look-alikes: a .ts file, a test-like name without .test., or a doc inside dist/', () => {
    cleanApp()
    put('src/lib/loader.ts', REAL)
    put('src/lib/attest.ts', REAL)
    put('src/lib/loader.test.json', REAL)
    put('scripts/guardDist.mjs.bak', REAL)
    put('dist/notes.md', REAL)
    expect(rules(checkBuild(root))).toEqual([
      'json-not-synthetic:src/lib/loader.test.json',
      'real-marker:dist/notes.md',
      'real-marker:scripts/guardDist.mjs.bak',
      'real-marker:src/lib/attest.ts',
      'real-marker:src/lib/loader.test.json',
      'real-marker:src/lib/loader.ts',
    ])
  })
})

describe('missing or empty dist/', () => {
  it('fails when dist/ does not exist, because nothing was scanned', () => {
    put('public/data/synthetic/manifest.json', SYNTHETIC)
    expect(rules(checkBuild(root))).toEqual(['dist-missing:dist'])
  })
  it('fails when dist/ exists but holds no files', () => {
    mkdirSync(join(root, 'dist/assets'), { recursive: true })
    expect(rules(checkBuild(root))).toEqual(['dist-empty:dist'])
  })
  it('fails when dist is a file, not a directory', () => {
    put('dist', 'not a folder')
    expect(rules(checkBuild(root))).toEqual(['dist-empty:dist', 'not-a-directory:dist'])
  })
})

describe('symlinks anywhere in the app tree', () => {
  /** Targets outside the app that hold real data. */
  const outside = () => ({ file: join(outsideRoot, 'real.json'), dir: outsideRoot })

  it('flags a symlink to a file inside public/', () => {
    cleanApp()
    symlinkSync(outside().file, join(root, 'public/data/link.json'))
    expect(rules(checkBuild(root))).toEqual(['symlink:public/data/link.json'])
  })
  it('flags a symlink to a directory, and a broken symlink, inside public/', () => {
    cleanApp()
    symlinkSync(outside().dir, join(root, 'public/dirlink'))
    symlinkSync(join(outsideRoot, 'no-such-target'), join(root, 'public/broken'))
    expect(rules(checkBuild(root))).toEqual(['symlink:public/broken', 'symlink:public/dirlink'])
  })
  it('flags public/ itself when it is a symlink', () => {
    put('dist/index.html', '<html></html>')
    symlinkSync(outside().dir, join(root, 'public'))
    expect(rules(checkBuild(root))).toEqual(['symlink:public'])
  })
  it('flags a symlink in dist/ without reading through it', () => {
    cleanApp()
    symlinkSync(outside().dir, join(root, 'dist/data/ext'))
    // Only the link is reported: the real file behind it lives outside dist/ and is not followed.
    expect(rules(checkBuild(root))).toEqual(['symlink:dist/data/ext'])
  })
  it('terminates on a symlink loop', () => {
    cleanApp()
    symlinkSync(join(root, 'public'), join(root, 'public/data/loop'))
    expect(rules(checkBuild(root))).toEqual(['symlink:public/data/loop'])
  })
  it('flags every symlink in src/, whatever its name: a folder link, a .json link and a link with no extension', () => {
    // A name filter that ran before the symlink test would skip "realdir" and "notes.txt" (review D, V3).
    cleanApp()
    const t = outside()
    symlinkSync(t.dir, join(root, 'src/data/realdir'))
    symlinkSync(t.file, join(root, 'src/data/real.json'))
    symlinkSync(t.file, join(root, 'src/data/notes.txt'))
    mkdirSync(join(root, 'src/lib'), { recursive: true })
    symlinkSync(t.file, join(root, 'src/lib/LINK'))
    expect(rules(checkBuild(root))).toEqual(['symlink:src/data/notes.txt', 'symlink:src/data/real.json', 'symlink:src/data/realdir', 'symlink:src/lib/LINK'])
  })
  it('flags a symlink outside src/, public/ and dist/ too (the whole app tree is scanned)', () => {
    cleanApp()
    const t = outside()
    mkdirSync(join(root, 'scripts'))
    symlinkSync(t.dir, join(root, 'realdata'))
    symlinkSync(t.file, join(root, 'scripts/data.json'))
    expect(rules(checkBuild(root))).toEqual(['symlink:realdata', 'symlink:scripts/data.json'])
  })
  it('does not flag a top-level node_modules that is a symlink (a monorepo layout), and does not read through it', () => {
    cleanApp()
    symlinkSync(outside().dir, join(root, 'node_modules')) // the target holds a real-data file
    expect(checkBuild(root).violations).toEqual([])
  })
})

describe('public/data/real and dist/data/real', () => {
  it('flags an empty public/data/real/ folder: it must not exist at all', () => {
    cleanApp()
    mkdirSync(join(root, 'public/data/real'))
    expect(rules(checkBuild(root))).toEqual(['real-dir:public/data/real'])
  })
  it('flags public/data/real when it is a file rather than a folder', () => {
    cleanApp()
    put('public/data/real', 'x')
    // It is also a non-JSON file in a data directory.
    expect(rules(checkBuild(root))).toEqual(['data-file-not-json:public/data/real', 'real-dir:public/data/real'])
  })
  it('flags public/data/real when it is a symlink, as a symlink and as the forbidden name', () => {
    cleanApp()
    symlinkSync(outsideRoot, join(root, 'public/data/real'))
    expect(rules(checkBuild(root))).toEqual(['real-dir:public/data/real', 'symlink:public/data/real'])
  })
  it('flags a folder full of real data in both the input and the output, and the marker in each file', () => {
    cleanApp()
    put('public/data/real/manifest.json', REAL)
    put('dist/data/real/manifest.json', REAL)
    expect(rules(checkBuild(root))).toEqual([
      'json-not-synthetic:dist/data/real/manifest.json',
      'json-not-synthetic:public/data/real/manifest.json',
      'real-dir:dist/data/real',
      'real-dir:public/data/real',
      'real-marker:dist/data/real/manifest.json',
      'real-marker:public/data/real/manifest.json',
    ])
  })
  it('flags dist/data/real even when its file carries no marker', () => {
    cleanApp()
    put('dist/data/real/blob.json', '{"data_mode":"synthetic"}')
    expect(rules(checkBuild(root))).toEqual(['real-dir:dist/data/real'])
  })
  it('does not flag a folder that merely starts with the name', () => {
    cleanApp()
    put('public/data/real-estate/x.json', SYNTHETIC)
    put('public/data/surreal/x.json', SYNTHETIC)
    expect(checkBuild(root).violations).toEqual([])
  })
})

describe('the positive allowlist: every JSON file must be synthetic, whether or not it carries the marker', () => {
  it('accepts a synthetic JSON file with any other fields', () => {
    cleanApp()
    put('src/data/more.json', '{"schema_version":1,"data_mode":"synthetic","tenors":["9D"],"levels":[[1.5]]}')
    expect(checkBuild(root).violations).toEqual([])
  })

  // Real data written by df.to_json() has no data_mode at all (review D, V5): the marker scan cannot see it.
  const REJECTED = [
    ['no data_mode field (real data without the marker)', '{"dates":["2024-08-05"],"close":[38.58]}'],
    ['data_mode with another value', '{"data_mode":"demo"}'],
    ['data_mode in the wrong case', '{"data_mode":"Synthetic"}'],
    ['data_mode null', '{"data_mode":null}'],
    ['data_mode only nested, not at the top level', '{"meta":{"data_mode":"synthetic"},"close":[38.58]}'],
    ['a top-level array', '[{"data_mode":"synthetic"}]'],
    ['a bare string', '"synthetic"'],
    ['not valid JSON', 'data_mode=synthetic'],
    ['an empty file', ''],
    ['a UTF-8 byte-order mark before the JSON (the parser refuses it)', '﻿{"data_mode":"synthetic"}'],
  ]
  it.each(REJECTED)('rejects a data file with %s', (_name, body) => {
    cleanApp()
    put('src/data/leak.json', body)
    expect(rules(checkBuild(root))).toEqual(['json-not-synthetic:src/data/leak.json'])
  })

  it('applies to every folder of src/, public/ and dist/, not only to data/', () => {
    cleanApp()
    put('src/lib/rates.json', '{"close":[1]}')
    put('public/vix.json', '{"close":[1]}')
    put('dist/assets/extra.json', '{"close":[1]}')
    expect(rules(checkBuild(root))).toEqual(['json-not-synthetic:dist/assets/extra.json', 'json-not-synthetic:public/vix.json', 'json-not-synthetic:src/lib/rates.json'])
  })

  it('treats the extension case-insensitively', () => {
    cleanApp()
    put('src/data/LEAK.JSON', '{"close":[1]}')
    expect(rules(checkBuild(root))).toEqual(['json-not-synthetic:src/data/LEAK.JSON'])
  })

  it('rejects any non-JSON file in a data directory, which closes CSV and gzip routes into the bundle', () => {
    cleanApp()
    put('src/data/vix.csv', 'date,close\n2024-08-05,38.58\n')
    put('public/data/term_structure.json.gz', 'gz-bytes')
    put('dist/data/blob.bin', 'bytes')
    put('src/data/loader.ts', 'export const x = 1')
    expect(rules(checkBuild(root))).toEqual([
      'data-file-not-json:dist/data/blob.bin',
      'data-file-not-json:public/data/term_structure.json.gz',
      'data-file-not-json:src/data/loader.ts',
      'data-file-not-json:src/data/vix.csv',
    ])
  })

  it('leaves non-JSON files outside the data directories alone', () => {
    cleanApp()
    put('public/robots.txt', 'User-agent: *')
    put('src/lib/format.ts', 'export {}')
    expect(checkBuild(root).violations).toEqual([])
  })

  it('does not look at JSON outside src/, public/ and dist/ (package.json, tsconfig.json): see DATA.md for this known limit', () => {
    cleanApp()
    put('package.json', '{"name":"app"}')
    put('tsconfig.json', '{"files":[]}')
    // Pinned on purpose: markerless real JSON placed here is NOT detected. If this is ever tightened, change this test.
    put('realdata/vix.json', '{"close":[38.58]}')
    expect(checkBuild(root).violations).toEqual([])
  })
})

describe('the command line', () => {
  const run = (...args) => spawnSync(process.execPath, [SCRIPT, ...args], { encoding: 'utf8' })

  it('exits 0 and says OK for a clean build', () => {
    cleanApp()
    const r = run('--root', root)
    expect(r.status).toBe(0)
    expect(r.stdout).toContain('guard:dist OK: scanned 3 file(s) in dist/ and 2 elsewhere under the app; 3 JSON file(s), all synthetic')
    expect(r.stderr).toBe('')
  })
  it('exits 1 and names the rule and the path when the marker is present', () => {
    cleanApp()
    put('dist/assets/x.js', REAL)
    const r = run('--root', root)
    expect(r.status).toBe(1)
    expect(r.stderr).toContain('guard:dist FAIL [real-marker]: dist/assets/x.js contains')
    expect(r.stderr).toContain('1 violation(s)')
    expect(r.stdout).toBe('')
  })
  it('exits 1 for a markerless JSON in a data directory', () => {
    cleanApp()
    put('src/data/vix.json', '{"close":[38.58]}')
    const r = run('--root', root)
    expect(r.status).toBe(1)
    expect(r.stderr).toContain('guard:dist FAIL [json-not-synthetic]: src/data/vix.json has no top-level "data_mode": "synthetic"')
  })
  it('inspects the app that holds the script when --root is not given, wherever it is run from', () => {
    // Copy the script into <root>/scripts/ so "the app that holds it" is the temp folder, then run
    // it from a different working directory: a wrong default root would look at the wrong place.
    mkdirSync(join(root, 'scripts'))
    copyFileSync(SCRIPT, join(root, 'scripts/guardDist.mjs'))
    cleanApp()
    const runCopy = () => spawnSync(process.execPath, [join(root, 'scripts/guardDist.mjs')], { encoding: 'utf8', cwd: tmpdir() })
    expect(runCopy().status).toBe(0)
    put('realdata/leak.json', REAL)
    const dirty = runCopy()
    expect(dirty.status).toBe(1)
    expect(dirty.stderr).toContain('realdata/leak.json')
  })
  it('exits 2 when --root has no value', () => {
    const r = run('--root')
    expect(r.status).toBe(2)
    expect(r.stderr).toContain('--root needs a directory')
  })
  it('exits 2 when --root is not a directory, instead of crashing', () => {
    const r = run('--root', join(root, 'no-such-folder'))
    expect(r.status).toBe(2)
    expect(r.stderr).toContain('does not exist or is not a directory')
  })
})
