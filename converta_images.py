import os

folder = "criminals"

for file in os.listdir(folder):
    if file.endswith(".jpeg"):
        old = os.path.join(folder, file)
        new = os.path.join(folder, file.replace(".jpeg", ".jpg"))
        os.rename(old, new)

print("Conversion completed")