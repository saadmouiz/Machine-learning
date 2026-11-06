import unittest
from pathlib import Path

from PIL import Image

from src.dataset import analyze_image_directory


class DatasetAnalysisTest(unittest.TestCase):
    def test_analyze_image_directory_counts_readable_and_unreadable_images(self):
        with self.subTest("temp dataset"):
            temp_dir = Path(".") / "tmp_dataset_test"
            if temp_dir.exists():
                for child in temp_dir.iterdir():
                    if child.is_file():
                        child.unlink()
            else:
                temp_dir.mkdir()

            valid_path = temp_dir / "ok.jpg"
            unreadable_path = temp_dir / "broken.jpg"

            image = Image.new("RGB", (32, 32), color="red")
            image.save(valid_path, format="JPEG")
            unreadable_path.write_bytes(b"not-a-valid-image")

            stats = analyze_image_directory(temp_dir)

            self.assertEqual(stats["total_images"], 2)
            self.assertEqual(stats["readable_images"], 1)
            self.assertEqual(stats["unreadable_images"], [unreadable_path])

            for child in temp_dir.iterdir():
                if child.is_file():
                    child.unlink()
            temp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
