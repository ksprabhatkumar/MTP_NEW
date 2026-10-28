from pathlib import Path
import sys
p=Path(sys.argv[1]); n=sys.argv[2]
with (p/"events.txt").open("a") as f: f.write(n+"\n")
if n=="2" and not (p/"retry_allowed").exists(): sys.exit(7)
(p/(n+".done")).write_text("complete")
