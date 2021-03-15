from aifeynman import run_aifeynman

# run_aifeynman("example_data/", "example1.txt", 30, "14ops.txt", polyfit_deg=3, NN_epochs=500, gpu=1)
# run_aifeynman("./dataset/Feynman_with_units/", "I.50.26", 60, "14ops.txt", polyfit_deg=3, NN_epochs=500, gpu=1)

# run_aifeynman("../aipeyman/dataset/Feynman_with_units/", "III.9.52", 60, "14ops.txt", polyfit_deg=3, NN_epochs=500)

run_aifeynman("example_data/", "example1.txt", 30, "14ops.txt", polyfit_deg=3, NN_epochs=1500, gpu=1)
