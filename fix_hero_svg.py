import re

filepath = "c:/Users/Shobhit Raj/Downloads/shobhit-jarvis-exact-readme/New folder/lucifer9973/assets/hero-dark.svg"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

open_cnt = content.count("<g ")
close_cnt = content.count("</g>")
diff = open_cnt - close_cnt
print(f"Open: {open_cnt}, Close: {close_cnt}, Diff: {diff}")

# Add closing tags before </svg>
closing_tags = "</g>\n" * diff
content = content.replace("</svg>", closing_tags + "</svg>")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

# Verify
open_cnt2 = content.count("<g ")
close_cnt2 = content.count("</g>")
print(f"After fix - Open: {open_cnt2}, Close: {close_cnt2}, Balanced: {open_cnt2 == close_cnt2}")
</create_file>
