def compare_output(o1: str, o2: str):
    for line in zip(o1.split("\n"), o2.split("\n")):
        print(f"{line[0]}{line[1]:>60}")


class JupyterPrinter:

    def __init__(self, print_output, detector_name):
        self.detector_name = detector_name
        self.print_output = print_output
        self.output = ""
        self.data_map = {}

    def add_experiment(self, name, columns=['error', 'keypoints', 'tentatives', 'inliers'], highlight=[True, False, False, False]):
        self.data_map[name] = {"columns": columns, "data": [], "highlight": highlight}
        self.current_map = self.data_map[name]
        self.current_map["special"] = {}

    def add_data(self, key_and_column_values):
        self.current_map["data"].append(key_and_column_values)

    def add_special_key(self, key, data):
        self.current_map["special"][key] = data

    def compare_in_table(self, other, use_columns=None):

        assert self.data_map.keys() == other.data_map.keys()
        for exp_key in self.data_map:

            print()
            print(f"#### Experiment: {exp_key}")
            map1 = self.data_map[exp_key]
            map2 = other.data_map[exp_key]

            # This assumes map1["columns"] is the same for all experiments (which is reasonable)
            if use_columns is None:
                use_columns = map1["columns"]

            use_columns_mask = [False] * len(map1["columns"])
            for col in use_columns:
                if map1["columns"].__contains__(col):
                    use_columns_mask[map1["columns"].index(col)] = True
            columns_len = sum(use_columns_mask)

            assert map1["special"].keys() == map2["special"].keys()
            assert map1["columns"] == map2["columns"]
            assert map1["highlight"] == map2["highlight"]

            for k in map1["special"]:
                print()
                print(f"* {k}")
                print()
                print(f"|{self.detector_name} | {other.detector_name}|")
                print(f"|---|--|")
                print(f"|{map1['special'][k]}|{map2['special'][k]}|")

            print()
            print("* Homography estimation")
            print(f"| | {'|'.join([f'{self.detector_name} | {other.detector_name}'] * columns_len)} |")
            print("|" + "|".join(["----"] * (2 * columns_len + 1)) + "|")
            print("|  | " + "|".join([f" {c}| " for i, c in enumerate(map1["columns"]) if use_columns_mask[i]]) + "|")

            decimal_point_present = [False] * (len(map2["data"][0]) - 1) * 2
            for i, data in enumerate(map1["data"]):
                assert data[0] == map2["data"][i][0]
                for j, value in enumerate(data[1:]):
                    if value.__contains__("."):
                        decimal_point_present[j * 2] = True
                    if map2["data"][i][j + 1].__contains__("."):
                        decimal_point_present[j * 2 + 1] = True

            for i, data in enumerate(map1["data"]):
                row = f"| {data[0]} | "
                for j, d in enumerate(data[1:]):

                    if not use_columns_mask[j]:
                        continue

                    def dec_point(el, dec, highlight):
                        if highlight:
                            el = f"__{el}__"
                        if dec and not el.__contains__("."):
                            return f"{el}&nbsp;&nbsp;&nbsp;"
                        else:
                            return el

                    def float_or_none(f):
                        try:
                            return float(f)
                        except Exception:
                            return None

                    def to_str(el1, decimal_point1, el2, decimal_point2, highlight):
                        if highlight:
                            f1 = float_or_none(el1)
                            f2 = float_or_none(el2)
                            if f1 is None or f2 is None:
                                highlight_loc = [not f1 is None, not f2 is None]
                            else:
                                highlight_loc = [float(el1) <= float(el2), float(el1) >= float(el2)]
                        else:
                            highlight_loc = [False, False]
                        return f" {dec_point(el1, decimal_point1, highlight_loc[0])} | {dec_point(el2, decimal_point2, highlight_loc[1])} | "

                    row = row + to_str(d, decimal_point_present[j * 2], map2['data'][i][j + 1], decimal_point_present[j * 2 + 1], map1["highlight"][j])

                print(row)

    def print(self, s=""):
        if self.print_output:
            print(s)
        self.output = self.output + s + "\n"

    def print_self(self):
        print(f"printer = JupyterPrinter(False, '{self.detector_name}')")
        print(f"printer.data_map = {self.data_map}")
