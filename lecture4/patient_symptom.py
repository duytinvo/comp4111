from patient_info import PatientInfo


class PatientSymptom(PatientInfo):
    disease = "FLU"
    main_symptoms = set(["runny-nose", "sinus", "cough", "fever", "muscle-ache"])

    def __init__(self, **patient):
        # PatientInfo.__init__(self, **patient)
        super(self.__class__, self).__init__(**patient)

    def print_info(self):
        print("=" * 40)
        print("\t\tPatient's INFORMATION: ")
        print(f"\t\t\t++ name: {self.name}")
        print(f"\t\t\t++ age: {self.age}")
        print(f"\t\t\t++ country: {self.country}")
        print(f"\t\t\t++ gender: {self.gender}")
        print("=" * 40)

    def deprecated_print_info(self):
        super().print_info()

    def diagnose(self, findings: list) -> bool:
        """
        This function is used to diagnose if a given patient getting flu or not
        :param findings: a list of symptoms from a given patient
        :return: True if findings contain at least 3 symptoms listed in main_symptoms else False
        """
        pass


if __name__ == '__main__':
    patient = PatientSymptom(name="Mary", age=23, gender="female")
    patient.print_info()
    patient.deprecated_print_info()
