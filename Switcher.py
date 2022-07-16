"""
# FINGER[0] THUMB
# FINGER[1] FIRST FINGER
"""

class PythonSwitch:
    def CheckShwonFinger(self, num, array):
        if num == 1:
            if array[1] == 1 and array[0] == 0 and array[2] == 0 and array[3] == 0 and array[4] == 0:
                return True
            else:
                return False
        elif num == 2:
            if array[1] == 1 and array[0] == 0 and array[2] == 1 and array[3] == 0 and array[4] == 0:
                return True
            else:
                return False
        elif num == 3:
            if array[1] == 1 and array[0] == 0 and array[2] == 1 and array[3] == 1 and array[4] == 0:
                return True
            else:
                return False
        elif num == 4:
            if array[1] == 1 and array[0] == 0 and array[2] == 1 and array[3] == 1 and array[4] == 1:
                return True
            else:
                return False
        elif num == 5:
            if array[1] == 1 and array[0] == 1 and array[2] == 1 and array[3] == 1 and array[4] == 1:
                return True
            else:
                return False