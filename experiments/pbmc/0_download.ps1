$url_pre = "https://raw.githubusercontent.com/PeterZZQ/scMoMaT/main/data/real/ASAP-PBMC/"
$save_dir = ".\data"

if (-not (Test-Path -Path $save_dir)) {
    New-Item -ItemType Directory -Path $save_dir
}

$files = @("genes.txt", "proteins.txt", "regions.txt", "GxR.npz")
for ($i = 1; $i -lt 3; $i++) {
    $files += "GxC$i.npz"
}
for ($i = 3; $i -lt 5; $i++) {
    $files += "RxC$i.npz"
}
for ($i = 1; $i -lt 5; $i++) {
    $files += "PxC$i.npz"
    $files += "meta_c$i.csv"
}

foreach ($filei in $files) {
    $save_path = Join-Path -Path $save_dir -ChildPath $filei
    Invoke-WebRequest -Uri "${url_pre}${filei}" -OutFile $save_path
}

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/luyiyun/mmAAVI/refs/heads/main/experiments/PBMC/data/proteins_alias.txt" -OutFile (Join-Path -Path $save_dir -ChildPath "proteins_alias.txt")
