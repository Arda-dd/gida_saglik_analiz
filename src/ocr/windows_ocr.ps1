# Windows'un yerlesik OCR motoru (Windows.Media.Ocr, Windows 10/11) - ek kurulum/indirme
# gerektirmez, sadece ilgili dilin Windows dil paketinde OCR bileseni olmasi yeterli.
# Cikti: kelime bazinda metin + sinir kutusu (JSON) - Python tarafi (src/ocr/extract.py
# extract_text_windows) Tesseract/EasyOCR ile ayni _group_boxes_into_rows mantigini uygular.
param(
    [Parameter(Mandatory = $true)][string]$Path,
    [string]$Lang = "tr"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
[void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
[void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
[void][Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]

# WinRT IAsyncOperation<T> -> .NET Task<T> (PowerShell 5.1 'await' bilmedigi icin)
$asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq "IAsyncOperation``1"
} | Select-Object -First 1

function Await($operation, [Type]$resultType) {
    $task = $asTaskGeneric.MakeGenericMethod($resultType).Invoke($null, @($operation))
    $task.Wait(-1) | Out-Null
    return $task.Result
}

$language = New-Object Windows.Globalization.Language($Lang)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) {
    throw "Windows OCR '$Lang' dilini desteklemiyor (Ayarlar > Dil > ilgili dilin OCR bileseni)"
}

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($Path)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$stream.Dispose()

$words = foreach ($line in $result.Lines) {
    foreach ($word in $line.Words) {
        [pscustomobject]@{
            text   = $word.Text
            x      = $word.BoundingRect.X
            y      = $word.BoundingRect.Y
            height = $word.BoundingRect.Height
        }
    }
}
ConvertTo-Json -InputObject @($words) -Compress
