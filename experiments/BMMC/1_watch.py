import subprocess


file_paths = [
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681518_MNC_RNA_counts.tsv',
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681519_MNC_ADT_counts.tsv',
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681520_MNC_HTO_counts.tsv',
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681521_MNC_BULK_RNA.tsv'
]

for file_path in file_paths:
    print(f"\nprocssing: {file_path}")


    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    num_rows = int(result.stdout.split()[0]) 


    result = subprocess.run(['head', '-n', '1', file_path], stdout=subprocess.PIPE, text=True)
    num_cols = len(result.stdout.strip().split('\t')) 


    print(f"table shape: ({num_rows}, {num_cols})")

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: 
                header = line.strip().split('\t')
                print(header[:5]) 
            else:
                data = line.strip().split('\t')
                print(data[:5])  

            if i >= 5:  
                break
