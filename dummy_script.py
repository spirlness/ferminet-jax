import subprocess
try:
    with open("src/ferminet/mcmc.py", "r") as f:
        code = f.read()

    if "key, subkey = _split_key(key)" in code:
        print("YES IT IS IN THE CODE")
    else:
        print("NO IT IS NOT IN THE CODE")
except Exception as e:
    print(e)
