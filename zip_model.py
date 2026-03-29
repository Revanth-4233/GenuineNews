import zipfile, os
with zipfile.ZipFile('model/detector.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('model/detector.pkl', 'detector.pkl')
print('Zip size:', os.path.getsize('model/detector.zip'))
