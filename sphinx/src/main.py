from module.inference import Inference
"""_summary_mainの記述
"""
class TestClass:
    """Summary line.
    """

    def testfunc(self, x, y):
        """sum

        Args:
            x (int): 1st argument
            y (int): 2nd argument

        Returns:
            int: sum result

        Examples:
            >>> print(testfunc(2,5))
            7
        """
        return x + y


def execute(a,b):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_テスト
    """    
    answer = Inference(a,b)
    return answer

if __name__ == '__main__':
    a = 1
    b = 2
    print(execute(a,b))



