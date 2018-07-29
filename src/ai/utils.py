def transform_board(board):
    def rescale(val):
        return val / 2

    input_data = [rescale(x) for y in board for x in y]
    return input_data