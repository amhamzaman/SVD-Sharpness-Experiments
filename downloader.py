import requests

folder = "D:/Stacks/DL Frames/13100/"

for i in range(500):
    print(f"downloading frame {i}")
    url = f"https://sip-ai-new.s3.amazonaws.com/fish_staging/329/880/1390/Frames/frame_{i}.jpg"
    response = requests.get(url)
    if response.status_code == 200:

        with open(f"{folder}frame_{i}.jpg", "wb") as f:
            f.write(response.content)
    else:
        print(f"Not found {response.status_code}")
        break

print("Done!")