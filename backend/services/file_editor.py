import os

class FileEditorService:
    """
    Service for surgical file modifications.
    """
    root_path: str

    def __init__(self, root_path: str) -> None:
        self.root_path = root_path

    def _safe_path(self, path: str) -> str:
        """
        Validate and return an absolute path within the root.
        """
        abs_root: str = os.path.abspath(self.root_path)
        safe_path: str = os.path.abspath(os.path.join(abs_root, path))
        if not safe_path.startswith(abs_root):
            raise ValueError(f'Path {path} is outside of root {abs_root}')
        return safe_path

    async def edit_file(self, path: str, search_block: str, replace_block: str) -> bool:
        """
        Apply a surgical edit to a file.

        :param path: Relative path to the file.
        :param search_block: The block of code to find.
        :param replace_block: The block of code to replace it with.
        :returns: True if the edit was successful, False otherwise.
        """
        safe_path: str = self._safe_path(path)
        if not os.path.exists(safe_path):
            return False

        content: str
        with open(safe_path, 'r') as f:
            content = f.read()

        if search_block not in content:
            return False

        new_content: str = content.replace(search_block, replace_block)
        
        with open(safe_path, 'w') as f:
            f.write(new_content)
            
        return True

    async def create_file(self, path: str, content: str) -> None:
        """
        Create a new file.

        :param path: Relative path to the file.
        :param content: The content of the file.
        """
        safe_path: str = self._safe_path(path)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, 'w') as f:
            f.write(content)
