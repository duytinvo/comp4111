class PatientInfo:
    course = "CDS@LU"

    def __init__(self, **patient):
        self.name = patient.pop("name", None)
        self.age = patient.pop("age", None)
        self.country = patient.pop("country", None)
        self.gender = patient.pop("gender", None)

    def print_info(self):
        print("-" * 40)
        print("\t\tPatient's INFO: ")
        print(f"\t\t\t++ name: {self.name}")
        print(f"\t\t\t++ age: {self.age}")
        print(f"\t\t\t++ country: {self.country}")
        print(f"\t\t\t++ gender: {self.gender}")
        print("-" * 40)
