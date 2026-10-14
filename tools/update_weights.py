#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Updates the packed weights (weights/*.vkoidnw) from a commit of RenderKit/oidn-weights.

For each model of the weights folder, downloads the .tza file (stored with Git LFS) at the given
commit, verifies its size and SHA-256, and packs it with pack_weights.py. The files are replaced
only if every model was downloaded and packed; a model with an unknown network structure stops
the update (vk_oidn then needs code changes, see weights/README.md). Then updates the source commit
in weights/README.md and, with --oidn-version, the Open Image Denoise release used as the reference
by the CI workflow.

Usage:
  update_weights.py <commit, tag or branch of oidn-weights> [--oidn-version X.Y.Z]
                    [--models rt_hdr,rt_hdr_alb] [--source <local oidn-weights checkout>] [--dry-run]

Only uses the Python standard library.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_weights  # noqa: E402

REPO = 'RenderKit/oidn-weights'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
README = os.path.join(WEIGHTS_DIR, 'README.md')
WORKFLOW = os.path.join(ROOT_DIR, '.github', 'workflows', 'build.yml')


def fetch(url, accept=None):
  request = urllib.request.Request(url, headers={'User-Agent': 'vk_oidn-update-weights'})
  if accept:
    request.add_header('Accept', accept)
  with urllib.request.urlopen(request, timeout=120) as response:
    return response.read()


def resolve_commit(ref):
  """Returns the full commit hash of a commit, tag or branch of oidn-weights."""
  if re.fullmatch(r'[0-9a-f]{40}', ref):
    return ref
  url = 'https://api.github.com/repos/%s/commits/%s' % (REPO, ref)
  try:
    return fetch(url, accept='application/vnd.github.sha').decode().strip()
  except urllib.error.HTTPError as e:
    if e.code in (404, 422):
      sys.exit('"%s" is not a commit, tag or branch of %s' % (ref, REPO))
    raise


def download_tza(commit, model, dest):
  """Downloads <model>.tza at the commit, verified against its Git LFS pointer."""
  path = '%s.tza' % model
  raw_url = 'https://raw.githubusercontent.com/%s/%s/%s' % (REPO, commit, path)
  try:
    pointer = fetch(raw_url)
  except urllib.error.HTTPError as e:
    if e.code == 404:
      raise ValueError('%s not found at this commit (removed or renamed?)' % path)
    raise

  if not pointer.startswith(b'version https://git-lfs'):
    data = pointer # stored directly in git
  else:
    fields = dict(line.split(' ', 1) for line in pointer.decode().splitlines() if ' ' in line)
    oid = fields['oid'].split(':', 1)[1]
    size = int(fields['size'])
    data = fetch('https://media.githubusercontent.com/media/%s/%s/%s' % (REPO, commit, path))
    if len(data) != size or hashlib.sha256(data).hexdigest() != oid:
      raise ValueError('%s: download does not match its Git LFS checksum' % path)

  with open(dest, 'wb') as f:
    f.write(data)


def local_commit(source):
  try:
    return subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return 'unknown'


def update_text(path, pattern, replacement, what):
  text = open(path, encoding='utf-8', newline='').read()
  new_text, count = re.subn(pattern, replacement, text, count=1)
  if count == 0:
    print('warning: could not update %s in %s' % (what, os.path.relpath(path, ROOT_DIR)))
    return False
  if new_text != text:
    open(path, 'w', encoding='utf-8', newline='').write(new_text)
  return True


def main():
  parser = argparse.ArgumentParser(description='Updates the packed weights from RenderKit/oidn-weights.')
  parser.add_argument('ref', help='commit, tag or branch of RenderKit/oidn-weights (e.g. a commit used by an OIDN release)')
  parser.add_argument('--oidn-version', help='Open Image Denoise release matching these weights (e.g. 2.6.0): '
                                             'updates weights/README.md and the CI reference')
  parser.add_argument('--models', help='comma-separated models (default: those in the weights folder)')
  parser.add_argument('--source', help='local oidn-weights checkout with the LFS files, instead of downloading')
  parser.add_argument('--dry-run', action='store_true', help='download and pack, but do not replace any file')
  args = parser.parse_args()

  if args.models:
    models = [m.strip() for m in args.models.split(',') if m.strip()]
  else:
    models = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(WEIGHTS_DIR, '*.vkoidnw')))
  if not models:
    sys.exit('no models to update')

  commit = local_commit(args.source) if args.source else resolve_commit(args.ref)
  print('oidn-weights commit %s, %d models' % (commit, len(models)))

  packed, errors = {}, []
  with tempfile.TemporaryDirectory() as tmp:
    for model in models:
      try:
        if args.source:
          tza = os.path.join(args.source, model + '.tza')
          with open(tza, 'rb') as f:
            if f.read(7) == b'version':
              raise ValueError('%s is a Git LFS pointer (run git lfs pull in %s)' % (tza, args.source))
        else:
          tza = os.path.join(tmp, model + '.tza')
          download_tza(commit, model, tza)
        packed[model] = pack_weights.pack(pack_weights.parse_tza(tza))
      except (ValueError, OSError, urllib.error.URLError) as e:
        errors.append('%s: %s' % (model, e))

  if errors:
    print('\nUpdate stopped, no file was changed:')
    for e in errors:
      print('  ' + e)
    if any('network structure' in e for e in errors):
      print('\nA model with an unknown network structure needs a new topology in tools/pack_weights.py\n'
            '(and its receptive field), matching core/unet_filter.cpp of Open Image Denoise.')
    return 1

  changed = []
  for model, blob in packed.items():
    path = os.path.join(WEIGHTS_DIR, model + '.vkoidnw')
    old = open(path, 'rb').read() if os.path.exists(path) else None
    status = 'new' if old is None else ('unchanged' if old == blob else 'changed')
    print('  %-26s %s (%d bytes)' % (model, status, len(blob)))
    if status != 'unchanged':
      changed.append((path, blob))

  if args.dry_run:
    print('\nDry run: %d file(s) would change, no file was written.' % len(changed))
    return 0

  for path, blob in changed:
    with open(path, 'wb') as f:
      f.write(blob)

  if re.fullmatch(r'[0-9a-f]{40}', commit):
    update_text(README, r'(commit\s+)`[0-9a-f]{40}`', r'\g<1>`%s`' % commit, 'the source commit')
  else:
    print('warning: unknown source commit, update it in weights/README.md')
  if args.oidn_version:
    update_text(README, r'\(Open Image Denoise [0-9.]+\)', '(Open Image Denoise %s)' % args.oidn_version,
                'the Open Image Denoise version')
    update_text(WORKFLOW, r'(OIDN_VERSION:\s*)[0-9.]+', r'\g<1>%s' % args.oidn_version, 'OIDN_VERSION')

  print('\n%d file(s) updated. Next steps:' % len(changed))
  print('  1. Rebuild (the weights are embedded in the library).')
  if not args.oidn_version:
    print('  2. Set OIDN_VERSION in .github/workflows/build.yml to the matching OIDN release '
          '(or rerun with --oidn-version).')
  print('  - Validate on the GPU with tests/scripts/run_validation.sh and the matching OIDN release as\n'
        '    OIDN_REF_DENOISE (delete the old ref_*.pfm files of the image folder first).')
  print('  - Push: the CI compares the denoised images with the OIDN release of OIDN_VERSION.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
