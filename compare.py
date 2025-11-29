import json

def load_json_file(filepath):
    """加载JSON文件"""
    with open(filepath, 'r') as f:
        return json.load(f)

def find_first_mismatch():
    """找出DC编码和解码数据首次失配的索引位置"""
    print("=== 查找DC数据首次失配位置 ===")
    
    # 加载数据
    dc_encoded = load_json_file('output/dc_encoded.json')
    dc_decoded = load_json_file('output/dc_decoded.json')
    
    encoded_data = dc_encoded['dc_encoded']
    decoded_data = dc_decoded['dc_encoded']
    
    print(f"DC编码数据数量: {len(encoded_data)}")
    print(f"DC解码数据数量: {len(decoded_data)}")
    
    # 逐个比较数据，找出首次失配的位置
    first_mismatch_idx = -1
    first_extra_in_decoded = -1  # 解码数据中首次出现编码数据中没有的数据的位置
    first_missing_in_decoded = -1  # 编码数据中首次出现解码数据中没有的数据的位置
    
    # 创建编码数据的集合用于快速查找
    encoded_set = set()
    encoded_list = []  # 保持顺序的列表
    for i, item in enumerate(encoded_data):
        key = (item['bits'], item['value'])
        encoded_set.add(key)
        encoded_list.append((i, key))
    
    # 创建解码数据的集合
    decoded_set = set()
    decoded_list = []  # 保持顺序的列表
    for i, item in enumerate(decoded_data):
        key = (item['bits'], item['value'])
        decoded_set.add(key)
        decoded_list.append((i, key))
    
    # 找出首次失配位置
    min_len = min(len(encoded_data), len(decoded_data))
    
    print("\n正在逐个比较数据...")
    for i in range(min_len):
        encoded_item = encoded_data[i]
        decoded_item = decoded_data[i]
        
        # 检查当前位置的数据是否匹配
        if (encoded_item['bits'] != decoded_item['bits'] or 
            encoded_item['value'] != decoded_item['value']):
            if first_mismatch_idx == -1:
                first_mismatch_idx = i
                print(f"🚨 首次失配位置: {i}")
                print(f"   编码数据: bits={encoded_item['bits']}, value={encoded_item['value']}")
                print(f"   解码数据: bits={decoded_item['bits']}, value={decoded_item['value']}")
    
    # 找出解码数据中首次出现编码数据中没有的数据
    print("\n查找解码数据中额外的数据...")
    for i, (idx, key) in enumerate(decoded_list):
        if key not in encoded_set:
            if first_extra_in_decoded == -1:
                first_extra_in_decoded = idx
                print(f"🚨 解码数据中首次出现编码数据中没有的数据: 位置 {idx}")
                print(f"   数据: bits={key[0]}, value={key[1]}")
                # 查看这个数据在编码数据中的情况
                print(f"   该数据在编码数据中出现的次数: {sum(1 for item in encoded_data if (item['bits'], item['value']) == key)}")
                break
    
    # 找出编码数据中首次出现解码数据中没有的数据
    print("\n查找编码数据中缺失的数据...")
    for i, (idx, key) in enumerate(encoded_list):
        if key not in decoded_set:
            if first_missing_in_decoded == -1:
                first_missing_in_decoded = idx
                print(f"🚨 编码数据中首次出现解码数据中没有的数据: 位置 {idx}")
                print(f"   数据: bits={key[0]}, value={key[1]}")
                # 查看这个数据在解码数据中的情况
                print(f"   该数据在解码数据中出现的次数: {sum(1 for item in decoded_data if (item['bits'], item['value']) == key)}")
                break
    
    # 更详细的分析：找出数据序列中的第一个差异点
    print("\n=== 详细序列分析 ===")
    encoded_idx = 0
    decoded_idx = 0
    
    # 用于追踪数据映射
    mappings = []  # [(encoded_idx, decoded_idx)]
    
    print("正在建立数据映射关系...")
    while (encoded_idx < len(encoded_data) and 
           decoded_idx < len(decoded_data) and 
           len(mappings) < 20):  # 限制分析前20个匹配项
        
        encoded_item = encoded_data[encoded_idx]
        decoded_item = decoded_data[decoded_idx]
        
        # 检查当前元素是否匹配
        if (encoded_item['bits'] == decoded_item['bits'] and 
            encoded_item['value'] == decoded_item['value']):
            # 匹配，记录映射关系
            mappings.append((encoded_idx, decoded_idx))
            encoded_idx += 1
            decoded_idx += 1
        else:
            # 不匹配，在解码数据中查找下一个匹配的编码元素
            found_at = -1
            for search_idx in range(decoded_idx + 1, min(decoded_idx + 30, len(decoded_data))):
                search_item = decoded_data[search_idx]
                if (encoded_item['bits'] == search_item['bits'] and 
                    encoded_item['value'] == search_item['value']):
                    found_at = search_idx
                    break
            
            if found_at != -1:
                # 找到了匹配项，中间的数据就是插入的数据
                print(f"🔍 在位置 {decoded_idx} 发现插入的数据，直到位置 {found_at-1}")
                for i in range(decoded_idx, found_at):
                    item = decoded_data[i]
                    print(f"   [{i}] bits={item['bits']}, value={item['value']}")
                
                # 记录映射关系
                mappings.append((encoded_idx, found_at))
                encoded_idx += 1
                decoded_idx = found_at + 1
            else:
                # 没找到匹配项，继续检查下一个
                decoded_idx += 1
    
    # 显示前几个映射关系
    print(f"\n前 {len(mappings)} 个数据映射关系:")
    for i, (enc_idx, dec_idx) in enumerate(mappings):
        enc_item = encoded_data[enc_idx]
        offset = dec_idx - enc_idx
        if offset == 0:
            print(f"  {i+1}. 编码[{enc_idx}] -> 解码[{dec_idx}]: ({enc_item['bits']},{enc_item['value']}) ✓ 正确位置")
        else:
            direction = "后移" if offset > 0 else "前移"
            print(f"  {i+1}. 编码[{enc_idx}] -> 解码[{dec_idx}]: ({enc_item['bits']},{enc_item['value']}) ⚠️  {direction}{abs(offset)}")

    # 总结结果
    print("\n=== 结果总结 ===")
    if first_mismatch_idx != -1:
        print(f"首次失配位置: 索引 {first_mismatch_idx}")
    else:
        print("✅ 前 {min_len} 项数据完全匹配")
    
    if first_extra_in_decoded != -1:
        print(f"解码数据中首次出现额外数据的位置: 索引 {first_extra_in_decoded}")
    
    if first_missing_in_decoded != -1:
        print(f"编码数据中首次缺失的数据位置: 索引 {first_missing_in_decoded}")
    
    return {
        'first_mismatch': first_mismatch_idx,
        'first_extra_in_decoded': first_extra_in_decoded,
        'first_missing_in_decoded': first_missing_in_decoded
    }

def show_detailed_comparison(start_idx, count=10):
    """显示指定位置的详细对比"""
    print(f"\n=== 详细对比 (索引 {start_idx} 开始的 {count} 项) ===")
    
    # 加载数据
    dc_encoded = load_json_file('output/dc_encoded.json')
    dc_decoded = load_json_file('output/dc_decoded.json')
    
    encoded_data = dc_encoded['dc_encoded']
    decoded_data = dc_decoded['dc_encoded']
    
    print(f"{'索引':<8} {'编码数据':<20} {'解码数据':<20} {'匹配状态':<10}")
    print("-" * 65)
    
    for i in range(start_idx, min(start_idx + count, len(encoded_data), len(decoded_data))):
        enc_item = encoded_data[i]
        dec_item = decoded_data[i]
        
        enc_str = f"({enc_item['bits']},{enc_item['value']})"
        dec_str = f"({dec_item['bits']},{dec_item['value']})"
        
        if enc_item['bits'] == dec_item['bits'] and enc_item['value'] == dec_item['value']:
            status = "✓"
        else:
            status = "✗"
        
        print(f"{i:<8} {enc_str:<20} {dec_str:<20} {status:<10}")

def main():
    """主函数"""
    print("开始查找DC编码和解码数据的首次失配位置...")
    
    # 查找首次失配位置
    result = find_first_mismatch()
    
    # 显示详细对比
    first_mismatch = result['first_mismatch']
    if first_mismatch != -1:
        # 在失配位置前后显示详细对比
        start_idx = max(0, first_mismatch - 5)
        show_detailed_comparison(start_idx, 15)
    else:
        # 如果没有失配，显示前几项的对比
        show_detailed_comparison(0, 15)

if __name__ == "__main__":
    main()