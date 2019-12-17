import shutil

import memory_access.memory_access as ma


class MemoryAccessTestExtension(ma.MemoryAccess):

    def delete_directory(self, graph_name: str) -> None:
        path = self._get_graph_base_path(graph_name=graph_name)
        shutil.rmtree(path)
