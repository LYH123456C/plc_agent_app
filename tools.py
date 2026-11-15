def check_scl_syntax(code: str) -> str:
    """
    检查生成的 SCL 代码是否符合基本语法（括号匹配、分号结尾等）。
    如果代码没问题，返回 "代码检查通过"。
    如果有问题，返回具体的错误信息。
    """
    errors = []
    
    # 1. 检查 BEGIN 和 END 是否成对
    if code.count("BEGIN") != code.count("END_ORGANIZATION_BLOCK") and code.count("BEGIN") != code.count("END_FUNCTION"):
        errors.append("错误：BEGIN 与 END_ORGANIZATION_BLOCK (或 END_FUNCTION) 数量不匹配。")
        
    # 2. 检查括号是否闭合
    if code.count("(") != code.count(")"):
        errors.append("错误：圆括号 '(' 和 ')' 数量不匹配，可能存在未闭合的逻辑。")
        
    # 3. 检查典型的 SCL 语句是否漏了分号
    # 3. 检查典型的 SCL 语句是否漏了分号
    lines = code.split("\n")
    for i, line in enumerate(lines):
        # 剥离注释部分再检查
        check_line = line.split("//")[0].strip()
        
        # 忽略空行和结构块
        if not check_line or check_line.startswith("BEGIN") or check_line.startswith("END"):
            continue
            
        if ":=" in check_line and not check_line.endswith(";"):
            errors.append(f"错误：第 {i+1} 行 '{check_line}' 漏了分号 ';'")
            if errors:
                return "\n".join(errors)
            else:
                return "代码语法检查通过，没有发现错误。"