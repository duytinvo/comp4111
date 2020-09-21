def diagnostic(disease, *symptoms, **patient):
    print("-" * 40)
    print(f"\t {disease.upper()} Diagnostic")
    print("-" * 40)
    print("\t\tPatient's Symptoms: ")
    for arg in symptoms:
        print(f"\t\t\t++ {arg}")
    print("-" * 40)
    print("\t\tPatient's INFO: ")
    for kw in patient:
        print(f"\t\t\t++ {kw}: {patient[kw]}")
    print("-" * 40)

