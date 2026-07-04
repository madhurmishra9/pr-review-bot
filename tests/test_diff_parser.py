from pr_review_bot.diff_parser import annotated_patch, parse_patch

PATCH = """\
@@ -1,4 +1,5 @@
 def add(a, b):
-    return a - b
+    result = a + b
+    return result

 def sub(a, b):
@@ -10,2 +11,3 @@ def mul(a, b):
     return a * b
+MULTIPLIER = 2
"""


def test_parse_added_lines():
    fd = parse_patch("calc.py", "modified", PATCH)
    assert fd.added_linenos == [2, 3, 12]
    assert fd.new_lines[2].content == "    result = a + b"
    assert fd.new_lines[3].content == "    return result"
    assert fd.new_lines[12].content == "MULTIPLIER = 2"


def test_parse_context_lines_are_not_additions():
    fd = parse_patch("calc.py", "modified", PATCH)
    assert fd.new_lines[1].content == "def add(a, b):"
    assert not fd.new_lines[1].is_addition
    assert fd.new_lines[11].content == "    return a * b"
    assert not fd.new_lines[11].is_addition


def test_removed_lines_tracked():
    fd = parse_patch("calc.py", "modified", PATCH)
    assert fd.removed_lines == ["    return a - b"]


def test_none_patch_yields_no_anchors():
    fd = parse_patch("image.png", "added", None)
    assert fd.new_lines == {}
    assert fd.added_linenos == []


def test_no_newline_marker_ignored():
    patch = "@@ -0,0 +1 @@\n+hello\n\\ No newline at end of file"
    fd = parse_patch("a.txt", "added", patch)
    assert fd.added_linenos == [1]
    assert fd.new_lines[1].content == "hello"


def test_annotated_patch_numbers_new_side():
    fd = parse_patch("calc.py", "modified", PATCH)
    text = annotated_patch(fd)
    assert "L2: +     result = a + b" in text
    assert "L12: + MULTIPLIER = 2" in text
    assert "      -     return a - b" in text
