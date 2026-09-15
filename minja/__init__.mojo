# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-minja/src/minja/__init__.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.com. Licensed under the Apache License, Version 2.0.
from .render import Renderer, render_chat
from .value import Heap, parse_json
