import os

def count_lines_of_code(folder_path, file_extension=".py"):
    total_lines = 0
    total_files = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    total_files += 1

    print(f"Total files: {total_files}")
    print(f"Total lines of code: {total_lines}")

# Example usage
count_lines_of_code(r"C:\Users\marci\Documents\GitHub\dyploma")  # replace with your actual path
