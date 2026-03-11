with open("src/ferminet/train.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "if not jnp.isfinite(energy_val):" in line:
        new_lines.append(line.replace("jnp.isfinite", "math.isfinite"))
        new_lines.insert(0, "import math\n")
    else:
        new_lines.append(line)

with open("src/ferminet/train.py", "w") as f:
    f.writelines(new_lines)
