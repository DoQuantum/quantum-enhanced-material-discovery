import pubchempy as pcp


def get_molecule_xyz(compound_name):
    """
    Fetches the 3D coordinates of a compound from PubChem and returns them in XYZ format.
    """
    try:
        compounds = pcp.get_compounds(compound_name, "name", record_type="3d")
        if not compounds:
            print(f"Could not find 3D structure for {compound_name} on PubChem.")
            return None

        # Take the first result
        compound = compounds[0]

        xyz_coords = []
        # The first line of XYZ is the number of atoms
        xyz_coords.append(str(len(compound.atoms)))
        # The second line is a comment, often the compound name
        xyz_coords.append(compound_name)

        for atom in compound.atoms:
            xyz_coords.append(f"{atom.element} {atom.x:.8f} {atom.y:.8f} {atom.z:.8f}")

        return "\n".join(xyz_coords)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    dbt_name = "Dibenzothiophene"
    dbt_xyz = get_molecule_xyz(dbt_name)

    if dbt_xyz:
        print(f"XYZ coordinates for {dbt_name}:\n")
        print(dbt_xyz)

        # Optionally, save to a file
        with open("dibenzothiophene.xyz", "w") as f:
            f.write(dbt_xyz)
        print("\nSaved coordinates to dibenzothiophene.xyz")
    else:
        print(f"Failed to retrieve coordinates for {dbt_name}.")
