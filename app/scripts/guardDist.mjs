/**
 * Deploy guard (ADR 0001, D2): fail if real market data could be shipped by this app.
 *   npm run guard:dist            (run by `npm run verify` after the build, and by the Dockerfile)
 *
 * Why it exists. `vite build` copies everything under public/ into dist/, and bundles anything the
 * code imports, from anywhere under the app directory. dist/ is what gets deployed. Real Cboe / FRED /
 * Yahoo data must never be in there (D2). The first line of defence is that real JSON is written
 * outside the repository; this script is the last line, and it fails loudly if the first one was
 * bypassed. It is a tripwire for accidents, not a defence against someone who deliberately hides
 * data (see "What it cannot catch" in DATA.md).
 *
 * It looks at the whole app directory (see `checkBuild`), except the top-level node_modules/,
 * coverage/ and .git/ folders, and fails on any of:
 *   1. A symlink anywhere. The test runs first, on every entry, so no file-name filter can skip a
 *      link (a link can point at data outside the repository, and `vite build` copies what it
 *      points at: measured, Vite 7.3.7). Links are never followed.
 *   2. The real-data marker `"data_mode":"real"` in any file, in dist/ or not. A bundler strips the
 *      literal from an imported JSON module (`{data_mode:X}` with `X="real"` kept elsewhere), so the
 *      dist/ scan cannot see it: it has to be found in the input, wherever the input lives. Files that
 *      only describe the marker (see MENTIONS_MARKER) are exempt.
 *   3. A positive allowlist, because real data may carry no marker at all (df.to_json): every .json
 *      file under src/, public/ and dist/ must have a top-level `"data_mode": "synthetic"`, and no
 *      file other than .json may sit in a data directory (src/data, public/data, dist/data).
 *   4. dist/ missing or empty: a guard that scanned nothing must not pass.
 *   5. public/data/real/ or dist/data/real/ existing at all, even empty.
 *
 * It never prints file contents, only paths and rule names: the thing it guards is the data. For the
 * same reason it never prints a JSON parse error, which quotes the text around the fault.
 */
import { lstatSync, readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

/**
 * `"data_mode": "real"` with any spacing. The pattern is looser than the literal on purpose:
 *  - `[\\"']*` before the colon also accepts no quotes at all (an object literal `{data_mode:"real"}`)
 *    and a backslash before the quote (JSON embedded in a JS string, `JSON.parse("{\"data_mode\":\"real\"}")`);
 *  - a closing quote is required straight after `real`, so "realistic" and "unreal" do not match.
 * ADR 0001 D2 and tests/test_contract.py spell the marker `"data_mode":"real"`; this is a superset.
 */
const REAL_MARKER = /data_mode[\\"']*\s*:\s*[\\"']*real[\\"']/

/** The app directory: this file lives in <app>/scripts/. */
const APP_ROOT = resolve(fileURLToPath(import.meta.url), '..', '..')

/**
 * Folders skipped, at the top level of the app only. A nested node_modules/ (say src/node_modules/)
 * is ordinary input: Vite resolves a relative import into it, so it is scanned like any other folder.
 */
const SKIPPED_TOP_LEVEL = new Set(['node_modules', 'coverage', '.git'])

/** Trees where every .json file must be synthetic: what the app can import (src/), serve (public/) or has built (dist/). */
const JSON_ROOTS = ['src', 'public', 'dist']

/** Data directories: nothing but synthetic .json files may live in them. */
const DATA_DIRS = ['src/data', 'public/data', 'dist/data']

/**
 * Files outside dist/ that talk about the marker without carrying data: the docs, the tests, and this
 * script (its comments and messages spell the marker). Exempt by path pattern, so a new test or doc
 * that quotes the marker needs no edit here; the cost is that real data hidden in such a file would
 * pass the marker scan (the data directories still reject anything that is not a synthetic .json).
 */
const MENTIONS_MARKER = (rel) => rel === 'scripts/guardDist.mjs' || /\.md$/i.test(rel) || /\.test\.(ts|tsx|mjs)$/.test(rel)

/**
 * Every file and symlink below `dir`, without following links (so a loop cannot hang the walk).
 * Paths are relative to `dir`, with `/` separators. Folders in `skipTopLevel` are not entered.
 */
function listTree(dir, skipTopLevel) {
  const entries = []
  const visit = (abs, rel) => {
    for (const e of readdirSync(abs, { withFileTypes: true })) {
      if (rel === '' && skipTopLevel.has(e.name)) continue
      const entryRel = rel === '' ? e.name : `${rel}/${e.name}`
      const entryAbs = join(abs, e.name)
      if (e.isSymbolicLink()) entries.push({ rel: entryRel, abs: entryAbs, kind: 'symlink' })
      else if (e.isDirectory()) visit(entryAbs, entryRel)
      else if (e.isFile()) entries.push({ rel: entryRel, abs: entryAbs, kind: 'file' })
      // sockets, FIFOs and devices are neither read nor reported: a FIFO would block the read
    }
  }
  visit(dir, '')
  return entries
}

/**
 * lstat that reports a missing path as null instead of throwing, and does not follow a final symlink.
 * ENOTDIR counts as missing: nothing can exist below a path component that is a file.
 */
function lstatOrNull(path) {
  try {
    return lstatSync(path)
  } catch (err) {
    if (err.code === 'ENOENT' || err.code === 'ENOTDIR') return null
    throw err
  }
}

function hasRealMarker(absPath) {
  // latin1 maps every byte to one character, so an ASCII marker is found in any file type
  // and a binary file cannot make the decoder throw.
  return REAL_MARKER.test(readFileSync(absPath, 'latin1'))
}

/**
 * Why a JSON file is not an acceptable data file, or null when it is: a top-level object whose
 * `data_mode` is exactly "synthetic". The top level is parsed, not searched with a regex, so
 * `{"meta":{"data_mode":"synthetic"},"v":[...]}` does not pass. The reason never quotes the file.
 */
function notSyntheticReason(absPath) {
  let doc
  try {
    doc = JSON.parse(readFileSync(absPath, 'utf8'))
  } catch {
    return 'is not valid JSON' // deliberately not the parser's message: it quotes the text
  }
  if (typeof doc !== 'object' || doc === null || Array.isArray(doc)) return 'is not a JSON object'
  if (doc.data_mode !== 'synthetic') return 'has no top-level "data_mode": "synthetic"'
  return null
}

/**
 * Inspect an app directory (the one holding package.json, dist/, public/ and src/).
 * Returns { violations, scanned }: violations is a list of { rule, path, message } and is empty
 * when the build is safe to ship; scanned counts the files read: `dist` and `app` (everything else)
 * for the marker scan, `json` for the JSON files checked against the allowlist.
 */
export function checkBuild(root) {
  const violations = []
  const add = (rule, path, message) => violations.push({ rule, path, message })
  const scanned = { dist: 0, app: 0, json: 0 }

  for (const e of listTree(root, SKIPPED_TOP_LEVEL)) {
    const path = e.rel
    // Symlinks first, for every entry, before any name or type filter (rule 1).
    if (e.kind === 'symlink') {
      add('symlink', path, `${path} is a symlink`)
      continue
    }
    const inDist = path.startsWith('dist/')

    // Rule 2: the marker, anywhere. dist/ has no exemptions; elsewhere only files that describe it.
    if (inDist || !MENTIONS_MARKER(path)) {
      scanned[inDist ? 'dist' : 'app'] += 1
      if (hasRealMarker(e.abs)) add('real-marker', path, `${path} contains "data_mode":"real"`)
    }

    // Rule 3: the positive allowlist.
    const isJson = path.toLowerCase().endsWith('.json')
    if (DATA_DIRS.some((dir) => path.startsWith(`${dir}/`)) && !isJson) {
      add('data-file-not-json', path, `${path} is in a data directory but is not a .json file`)
    } else if (isJson && JSON_ROOTS.includes(path.split('/')[0])) {
      scanned.json += 1
      const reason = notSyntheticReason(e.abs)
      if (reason !== null) add('json-not-synthetic', path, `${path} ${reason}`)
    }
  }

  // Rule 4.
  const distStat = lstatOrNull(join(root, 'dist'))
  if (distStat === null) add('dist-missing', 'dist', 'dist/ does not exist: run `npm run build` first, nothing was scanned')
  else if (!distStat.isSymbolicLink() && !distStat.isDirectory()) add('not-a-directory', 'dist', 'dist exists but is not a directory')
  if (distStat !== null && scanned.dist === 0) add('dist-empty', 'dist', 'dist/ contains no files, so nothing was scanned')

  // Rule 5.
  for (const dir of ['public/data/real', 'dist/data/real']) {
    if (lstatOrNull(join(root, dir)) !== null) add('real-dir', dir, `${dir} exists: real data must live outside the repository`)
  }
  return { violations, scanned }
}

/**
 * Command line entry. `--root <dir>` points at a different app directory (the tests use it).
 * Exit codes: 0 safe, 1 violation found, 2 bad usage. `log` is injectable so tests can capture it.
 */
export function main(argv, log = console) {
  let root = APP_ROOT
  const flag = argv.indexOf('--root')
  if (flag !== -1) {
    if (flag + 1 >= argv.length) {
      log.error('guard:dist: --root needs a directory')
      return 2
    }
    root = resolve(argv[flag + 1])
  }
  if (!lstatOrNull(root)?.isDirectory()) {
    log.error('guard:dist: the app directory to check does not exist or is not a directory')
    return 2
  }
  const { violations, scanned } = checkBuild(root)
  if (violations.length > 0) {
    for (const v of violations) log.error(`guard:dist FAIL [${v.rule}]: ${v.message}`)
    log.error(`guard:dist: ${violations.length} violation(s). Do not deploy this build (ADR 0001 D2).`)
    return 1
  }
  log.log(
    `guard:dist OK: scanned ${scanned.dist} file(s) in dist/ and ${scanned.app} elsewhere under the app; ` +
      `${scanned.json} JSON file(s), all synthetic; no real-data marker, no symlink, no public/data/real/`,
  )
  return 0
}

// Run only when executed as a script (`node scripts/guardDist.mjs`), not when a test imports it.
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exitCode = main(process.argv.slice(2))
}
