import pandas as pd
from scripts.review_errors import select_errors


def test_select_errors():
    frame = pd.DataFrame({
        'path': ['a.jpg', 'b.jpg', 'c.jpg'],
        'label': [0, 1, 0],
        'predicted_label': [1, 1, 0],
        'tumor_probability': [.8, .9, .1],
    })
    errors = select_errors(frame)
    assert errors['path'].tolist() == ['a.jpg']
