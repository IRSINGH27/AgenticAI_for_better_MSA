from typing import Any, List, Dict, Union


_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _is_primitive(v: Any) -> bool:
    return isinstance(v, _PRIMITIVE_TYPES)


def _needs_quoting(s: str) -> bool:
    if s == "":
        return True
    if s in ("true", "false", "null"):
        return True
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        pass
    structural_chars = [',', '\n', '\r', '\t', '[', ']', '{', '}', ':', '"', '\\']
    for c in structural_chars:
        if c in s:
            return True
    if s[0] in (' ', '-', '+') or s[-1] == ' ':
        return True
    return False


def _encode_string(s: str) -> str:
    if not _needs_quoting(s):
        return s
    escaped = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return f'"{escaped}"'


def _decode_key(key: str) -> str:
    key = key.strip()
    if len(key) >= 2 and key[0] == '"' and key[-1] == '"':
        inner = key[1:-1]
        inner = inner.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
        inner = inner.replace('\\"', '"').replace('\\\\', '\\')
        return inner
    return key


def _encode_primitive(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "null"
        if v == float('inf') or v == float('-inf'):
            return "null"
        return str(v)
    if isinstance(v, str):
        return _encode_string(v)
    raise TypeError(f"Cannot encode {type(v).__name__} as primitive")


def _is_uniform_primitive_array(arr: List[Any]) -> bool:
    if not arr or not all(isinstance(x, dict) for x in arr):
        return False
    first_keys = set(arr[0].keys())
    for item in arr[1:]:
        if set(item.keys()) != first_keys:
            return False
        for v in item.values():
            if not _is_primitive(v):
                return False
    return True


def _encode_array_inline(arr: List[Any]) -> str:
    parts = [_encode_primitive(v) for v in arr]
    return "[" + ",".join(parts) + "]"


def _encode_tabular(name: str, arr: List[Dict[str, Any]], indent: int, current_depth: int) -> List[str]:
    lines = []
    fields = list(arr[0].keys())
    header = f"{name}[{len(arr)}]{{{','.join(fields)}}}:"
    lines.append("  " * current_depth + header)
    inner_indent = "  " * (current_depth + 1)
    for item in arr:
        row_values = [_encode_primitive(item.get(f)) for f in fields]
        lines.append(inner_indent + ",".join(row_values))
    return lines


def _encode_list_form(name: str, arr: List[Any], indent: int, current_depth: int) -> List[str]:
    lines = []
    lines.append("  " * current_depth + f"{name}[{len(arr)}]:")
    inner_indent = "  " * (current_depth + 1)
    for item in arr:
        if isinstance(item, dict):
            first = True
            for k, v in item.items():
                prefix = "- " if first else "  "
                first = False
                if isinstance(v, dict):
                    if not v:
                        lines.append(inner_indent + f"{prefix}{k}:")
                    else:
                        lines.append(inner_indent + f"{prefix}{k}:")
                        lines.extend(_encode_object_body(v, indent, current_depth + 2))
                elif isinstance(v, list):
                    lines.extend(_encode_array_field(k, v, indent, current_depth + 1, is_list_item=True))
                else:
                    lines.append(inner_indent + f"{prefix}{k}: {_encode_primitive(v)}")
        else:
            lines.append(inner_indent + f"- {_encode_primitive(item)}")
    return lines


def _encode_array_field(name: str, arr: List[Any], indent: int, current_depth: int, is_list_item: bool = False) -> List[str]:
    if _is_uniform_primitive_array(arr):
        lines = []
        fields = list(arr[0].keys())
        prefix = "- " if is_list_item else ""
        header = f"{prefix}{name}[{len(arr)}]{{{','.join(fields)}}}"
        if is_list_item:
            lines.append("  " * current_depth + header + ":")
            inner_indent = "  " * (current_depth + 1)
        else:
            lines.append("  " * current_depth + header + ":")
            inner_indent = "  " * (current_depth + 1)
        for item in arr:
            row_values = [_encode_primitive(item.get(f)) for f in fields]
            lines.append(inner_indent + ",".join(row_values))
        return lines
    return _encode_list_form(name, arr, indent, current_depth)


def _encode_object_body(obj: Dict[str, Any], indent: int, current_depth: int) -> List[str]:
    lines = []
    for k, v in obj.items():
        encoded_key = _encode_string(k) if _needs_quoting(k) else k
        if isinstance(v, dict):
            if not v:
                lines.append("  " * current_depth + f"{encoded_key}:")
            else:
                lines.append("  " * current_depth + f"{encoded_key}:")
                lines.extend(_encode_object_body(v, indent, current_depth + 1))
        elif isinstance(v, list):
            if not v:
                lines.append("  " * current_depth + f"{encoded_key}[]:")
            elif all(_is_primitive(x) for x in v):
                lines.append("  " * current_depth + f"{encoded_key}{_encode_array_inline(v)}")
            else:
                lines.extend(_encode_array_field(encoded_key, v, indent, current_depth))
        else:
            lines.append("  " * current_depth + f"{encoded_key}: {_encode_primitive(v)}")
    return lines


def to_toon(obj: Any, indent: int = 2) -> str:
    if isinstance(obj, dict):
        if not obj:
            return ""
        lines = _encode_object_body(obj, indent, 0)
        return "\n".join(lines) + "\n"
    if isinstance(obj, list):
        if not obj:
            return "[]\n"
        if all(_is_primitive(x) for x in obj):
            return _encode_array_inline(obj) + "\n"
        if all(isinstance(x, dict) for x in obj):
            return _encode_root_list(obj, indent)
        return _encode_array_inline([_encode_primitive(x) for x in obj]) + "\n"
    return _encode_primitive(obj) + "\n"


def _encode_root_list(arr: List[Any], indent: int) -> str:
    if _is_fully_primitive_object_array(arr):
        fields = list(arr[0].keys())
        lines = [f"[{len(arr)}]{{{','.join(fields)}}}:"]
        for item in arr:
            row_values = [_encode_primitive(item.get(f)) for f in fields]
            lines.append(",".join(row_values))
        return "\n".join(lines) + "\n"

    lines = []
    for item in arr:
        if not isinstance(item, dict):
            lines.append(f"- {_encode_primitive(item)}")
            continue
        first = True
        for k, v in item.items():
            encoded_key = _encode_string(k) if _needs_quoting(k) else k
            prefix = "- " if first else "  "
            first = False
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{prefix}{encoded_key}:")
                else:
                    lines.append(f"{prefix}{encoded_key}:")
                    lines.extend(_encode_object_body(v, indent, 2))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{prefix}{encoded_key}[]:")
                elif all(_is_primitive(x) for x in v):
                    lines.append(f"{prefix}{encoded_key}{_encode_array_inline(v)}")
                elif _is_uniform_primitive_array(v):
                    fields = list(v[0].keys())
                    lines.append(f"{prefix}{encoded_key}[{len(v)}]{{{','.join(fields)}}}:")
                    for row in v:
                        row_values = [_encode_primitive(row.get(f)) for f in fields]
                        lines.append("    " + ",".join(row_values))
                else:
                    lines.append(f"{prefix}{encoded_key}[{len(v)}]:")
                    lines.extend(_encode_list_body(v, indent, 2))
            else:
                lines.append(f"{prefix}{encoded_key}: {_encode_primitive(v)}")
    return "\n".join(lines) + "\n"


def _is_fully_primitive_object_array(arr):
    if not arr or not all(isinstance(x, dict) for x in arr):
        return False
    for item in arr:
        if not all(_is_primitive(v) for v in item.values()):
            return False
    return True


def _encode_list_body(arr: List[Any], indent: int, current_depth: int) -> List[str]:
    lines = []
    inner = "  " * current_depth
    for item in arr:
        if not isinstance(item, dict):
            lines.append(inner + f"- {_encode_primitive(item)}")
            continue
        first = True
        for k, v in item.items():
            encoded_key = _encode_string(k) if _needs_quoting(k) else k
            prefix = "- " if first else "  "
            first = False
            if isinstance(v, dict):
                if not v:
                    lines.append(inner + f"{prefix}{encoded_key}:")
                else:
                    lines.append(inner + f"{prefix}{encoded_key}:")
                    lines.extend(_encode_object_body(v, indent, current_depth + 2))
            elif isinstance(v, list):
                if not v:
                    lines.append(inner + f"{prefix}{encoded_key}[]:")
                elif all(_is_primitive(x) for x in v):
                    lines.append(inner + f"{prefix}{encoded_key}{_encode_array_inline(v)}")
                elif _is_uniform_primitive_array(v):
                    fields = list(v[0].keys())
                    lines.append(inner + f"{prefix}{encoded_key}[{len(v)}]{{{','.join(fields)}}}:")
                    for row in v:
                        row_values = [_encode_primitive(row.get(f)) for f in fields]
                        lines.append(inner + "  " + ",".join(row_values))
                else:
                    lines.append(inner + f"{prefix}{encoded_key}[{len(v)}]:")
                    lines.extend(_encode_list_body(v, indent, current_depth + 1))
            else:
                lines.append(inner + f"{prefix}{encoded_key}: {_encode_primitive(v)}")
    return lines


class TOONDecodeError(ValueError):
    pass


def _parse_primitive(token: str):
    token = token.strip()
    if token == "null" or token == "":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if token.startswith('"') and token.endswith('"'):
        inner = token[1:-1]
        return inner.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
    try:
        if '.' in token or 'e' in token or 'E' in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _tokenize_row(line: str) -> List[str]:
    tokens = []
    current = []
    in_quotes = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '\\' and i + 1 < len(line):
            current.append(c + line[i+1])
            i += 2
            continue
        if c == '"':
            in_quotes = not in_quotes
            current.append(c)
            i += 1
            continue
        if c == ',' and not in_quotes:
            tokens.append(''.join(current))
            current = []
            i += 1
            continue
        current.append(c)
        i += 1
    if current:
        tokens.append(''.join(current))
    return tokens


def _parse_array_header(line: str):
    bracket_start = line.find('[')
    if bracket_start == -1:
        raise TOONDecodeError(f"Invalid array header: {line!r}")
    bracket_end = line.find(']', bracket_start)
    if bracket_end == -1:
        raise TOONDecodeError(f"Unclosed bracket in header: {line!r}")
    length_str = line[bracket_start+1:bracket_end]
    try:
        length = int(length_str)
    except ValueError:
        raise TOONDecodeError(f"Invalid array length: {length_str!r}")

    fields = None
    brace_start = line.find('{', bracket_end)
    if brace_start != -1:
        brace_end = line.find('}', brace_start)
        if brace_end == -1:
            raise TOONDecodeError(f"Unclosed brace in header: {line!r}")
        fields_str = line[brace_start+1:brace_end]
        fields = _tokenize_row(fields_str)
        name = line[:bracket_start].strip()
    else:
        name = line[:bracket_start].strip()

    colon_pos = line.find(':', max(bracket_end, brace_start + 1 if brace_start != -1 else 0))
    inline_values = None
    if colon_pos == -1 and '[' in line and ']' in line:
        after_bracket = line[bracket_end+1:].strip()
        if after_bracket.startswith('['):
            end = after_bracket.rfind(']')
            if end != -1:
                inline_values = _tokenize_row(after_bracket[1:end])

    return name, length, fields, inline_values


def from_toon(text: str) -> Any:
    if not text or not text.strip():
        return {}

    lines = text.splitlines()
    pos = [0]

    def _find_top_colon(s):
        in_quotes = False
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and i + 1 < len(s):
                i += 2
                continue
            if c == '"':
                in_quotes = not in_quotes
            elif c == ':' and not in_quotes:
                return i
            i += 1
        return -1

    def parse_object(min_indent):
        obj = {}
        while pos[0] < len(lines):
            line = lines[pos[0]]
            if not line.strip():
                pos[0] += 1
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent < min_indent:
                break
            if current_indent > min_indent:
                break
            stripped = line.strip()
            if stripped.startswith('- '):
                break
            if '[' in stripped and ']' in stripped and '{' in stripped and stripped.endswith(':'):
                bracket_pos = stripped.find('[')
                key = _decode_key(stripped[:bracket_pos])
                obj[key] = parse_nested_array(line, current_indent)
                continue
            colon_pos = _find_top_colon(stripped)
            if colon_pos == -1 and '[' in stripped and stripped.endswith(']'):
                bracket_pos_inline = stripped.find('[')
                key = _decode_key(stripped[:bracket_pos_inline])
                inner = stripped[bracket_pos_inline+1:-1]
                obj[key] = [_parse_primitive(t) for t in _tokenize_row(inner)]
                pos[0] += 1
                continue
            if colon_pos == -1:
                raise TOONDecodeError(f"Missing colon at line {pos[0]+1}: {stripped!r}")
            key = _decode_key(stripped[:colon_pos])
            rest = stripped[colon_pos+1:].strip()
            pos[0] += 1
            if rest == "":
                if pos[0] < len(lines):
                    next_line = lines[pos[0]]
                    if not next_line.strip():
                        obj[key] = {}
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent > min_indent:
                        stripped_next = next_line.strip()
                        if stripped_next.startswith('- '):
                            obj[key] = parse_list_value(next_indent)
                        elif '[' in stripped_next and ']' in stripped_next and '{' in stripped_next and stripped_next.endswith(':'):
                            obj[key] = parse_nested_array(next_line, next_indent)
                        elif '[' in stripped_next and ']' in stripped_next and stripped_next.endswith(':'):
                            obj[key] = parse_nested_array(next_line, next_indent)
                        else:
                            obj[key] = parse_object_value(next_indent)
                    else:
                        obj[key] = {}
                else:
                    obj[key] = {}
            elif rest.startswith('[') and rest.endswith(']'):
                inner = rest[1:-1]
                if not inner:
                    obj[key] = []
                else:
                    obj[key] = [_parse_primitive(t) for t in _tokenize_row(inner)]
            else:
                obj[key] = _parse_primitive(rest)
        return obj

    def parse_list_value(min_indent, expected_length=None):
        items = []
        last_pos = -1
        while pos[0] < len(lines):
            if pos[0] == last_pos:
                raise TOONDecodeError(f"parse_list_value: no progress at pos={pos[0]}: {lines[pos[0]]!r}")
            last_pos = pos[0]
            line = lines[pos[0]]
            if not line.strip():
                pos[0] += 1
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent < min_indent:
                break
            stripped = line.strip()
            if current_indent == min_indent:
                if not stripped.startswith('- '):
                    break
                pos[0] += 1
                first_content = stripped[2:]
                colon_pos = _find_top_colon(first_content)
                if colon_pos == -1:
                    items.append(_parse_primitive(first_content))
                    continue
                key = _decode_key(first_content[:colon_pos])
                rest = first_content[colon_pos+1:].strip()
                item = {}
                if rest:
                    item[key] = _parse_primitive(rest)
                else:
                    if pos[0] < len(lines):
                        next_line = lines[pos[0]]
                        if next_line.strip():
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent > current_indent:
                                stripped_next = next_line.strip()
                                if '[' in stripped_next and ']' in stripped_next and '{' in stripped_next and stripped_next.endswith(':'):
                                    item[key] = parse_nested_array(next_line, next_indent)
                                elif '[' in stripped_next and ']' in stripped_next and stripped_next.endswith(':'):
                                    item[key] = parse_nested_array(next_line, next_indent)
                                else:
                                    sub = parse_object_value(next_indent)
                                    if isinstance(sub, dict):
                                        item.update(sub)
                                    else:
                                        item[key] = {}
                            else:
                                item[key] = {}
                        else:
                            item[key] = {}
                    else:
                        item[key] = {}
                items.append(item)
            elif current_indent > min_indent and items:
                last_item = items[-1]
                if isinstance(last_item, dict):
                    stripped_line = line.strip()
                    if '[' in stripped_line and ']' in stripped_line and '{' in stripped_line and stripped_line.endswith(':'):
                        bracket_pos = stripped_line.find('[')
                        key = _decode_key(stripped_line[:bracket_pos])
                        last_item[key] = parse_nested_array(line, current_indent)
                    else:
                        sub = parse_object_value(current_indent)
                        if isinstance(sub, dict):
                            last_item.update(sub)
            else:
                break
            if expected_length is not None and len(items) >= expected_length:
                break
        return items

    def parse_object_value(min_indent):
        obj = {}
        while pos[0] < len(lines):
            line = lines[pos[0]]
            if not line.strip():
                pos[0] += 1
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent < min_indent:
                break
            if current_indent > min_indent:
                break
            stripped = line.strip()
            if stripped.startswith('- '):
                break
            if '[' in stripped and ']' in stripped and '{' in stripped and stripped.endswith(':'):
                bracket_pos = stripped.find('[')
                key = _decode_key(stripped[:bracket_pos])
                obj[key] = parse_nested_array(line, current_indent)
                continue
            bracket_pos_inline = stripped.find('[')
            colon_pos = _find_top_colon(stripped)
            if colon_pos == -1 and bracket_pos_inline != -1 and stripped.endswith(']'):
                key = _decode_key(stripped[:bracket_pos_inline])
                inner = stripped[bracket_pos_inline+1:-1]
                obj[key] = [_parse_primitive(t) for t in _tokenize_row(inner)]
                pos[0] += 1
                continue
            if colon_pos == -1:
                break
            key = _decode_key(stripped[:colon_pos])
            rest = stripped[colon_pos+1:].strip()
            pos[0] += 1
            if rest == "":
                if pos[0] < len(lines):
                    next_line = lines[pos[0]]
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent > min_indent:
                            if '[' in next_line and ']' in next_line and '{' in next_line and next_line.strip().endswith(':'):
                                obj[key] = parse_nested_array(next_line, next_indent)
                            elif '[' in next_line and ']' in next_line and next_line.strip().endswith(':'):
                                obj[key] = parse_nested_array(next_line, next_indent)
                            else:
                                obj[key] = parse_object_value(next_indent)
                        else:
                            obj[key] = {}
                    else:
                        obj[key] = {}
                else:
                    obj[key] = {}
            elif rest.startswith('[') and rest.endswith(']'):
                inner = rest[1:-1]
                if not inner:
                    obj[key] = []
                else:
                    obj[key] = [_parse_primitive(t) for t in _tokenize_row(inner)]
            else:
                obj[key] = _parse_primitive(rest)
        return obj

    def parse_nested_array(line, min_indent):
        stripped = line.strip()
        if '[' in stripped and ']' in stripped and '{' in stripped:
            name, length, fields, _ = _parse_array_header(stripped)
            pos[0] += 1
            if fields is not None:
                return _parse_tabular_array(lines, pos, min_indent, fields, length)
            if pos[0] < len(lines):
                return parse_list_value(min_indent + 2, length)
            return []
        if '[' in stripped and ']' in stripped:
            bracket_start = stripped.find('[')
            bracket_end = stripped.find(']', bracket_start)
            inside = stripped[bracket_start+1:bracket_end]
            if not inside:
                return []
            if '{' not in inside:
                return [_parse_primitive(t) for t in _tokenize_row(inside)]
        return []

    first_line = lines[0]
    first_indent = 0
    while pos[0] < len(lines) and not lines[pos[0]].strip():
        pos[0] += 1
    if pos[0] >= len(lines):
        return {}

    first_line = lines[pos[0]]
    first_stripped = first_line.strip()
    first_indent = len(first_line) - len(first_line.lstrip())

    has_primitive_marker = (':' not in first_stripped and
                             '[' not in first_stripped and
                             not first_stripped.startswith('- '))
    if has_primitive_marker:
        has_more = False
        for i in range(pos[0] + 1, len(lines)):
            if lines[i].strip():
                has_more = True
                break
        if not has_more:
            return _parse_primitive(first_stripped)

    if first_stripped.startswith('[') and not any(c.isalpha() for c in first_stripped.split('[')[0]):
        bracket_start = first_stripped.find('[')
        bracket_end = first_stripped.rfind(']')
        inside = first_stripped[bracket_start+1:bracket_end]
        if '{' in first_stripped and '}' in first_stripped:
            name, length, fields, _ = _parse_array_header(first_stripped)
            pos[0] += 1
            if fields is not None:
                return _parse_tabular_array(lines, pos, first_indent, fields, length)
            return parse_list_value(first_indent + 2, length)
        if not inside:
            return []
        tokens = _tokenize_row(inside)
        return [_parse_primitive(t) for t in tokens]

    if first_stripped.startswith('- '):
        return parse_list_value(first_indent)

    if '[' in first_stripped and ']' in first_stripped and not first_stripped.endswith(':'):
        bracket_start = first_stripped.find('[')
        bracket_end = first_stripped.rfind(']')
        inside = first_stripped[bracket_start+1:bracket_end]
        if not inside:
            return []
        tokens = _tokenize_row(inside)
        return [_parse_primitive(t) for t in tokens]

    pos[0] = 0
    return parse_object(0)


def _parse_tabular_array(lines, pos, min_indent, fields, expected_length):
    rows = []
    while pos[0] < len(lines) and len(rows) < expected_length:
        line = lines[pos[0]]
        if not line.strip():
            pos[0] += 1
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent < min_indent:
            break
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.endswith(':'):
            break
        tokens = _tokenize_row(stripped)
        if len(tokens) != len(fields):
            raise TOONDecodeError(f"Row at line {pos[0]+1} has {len(tokens)} fields, expected {len(fields)}")
        row = {f: _parse_primitive(t) for f, t in zip(fields, tokens)}
        rows.append(row)
        pos[0] += 1
    return rows


def _parse_root_list(lines, pos):
    return []


def _parse_list_item_body(lines, pos, min_indent):
    return {}


def parse_object_inline(min_indent):
    return {}
