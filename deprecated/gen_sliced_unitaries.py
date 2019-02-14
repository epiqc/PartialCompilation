import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="The filename to output to")
parser.add_argument("thetas", help="The parameters for the ansatz", nargs="*", type=complex)

args = parser.parse_args()
print(args.filename)
print(args.thetas)


from uccsd_unitary import save_uccsd_slices

save_uccsd_slices(args.thetas, args.filename)
