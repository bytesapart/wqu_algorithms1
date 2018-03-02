from PIL import Image, ImageDraw


def get_website_dataset():
    """
    This function fetches me the website data that I need

    :return: list
    """
    return [['slashdot', 'USA', 'yes', 18, 'None'],
            ['google', 'France', 'yes', 23, 'Premium'],
            ['digg', 'USA', 'yes', 24, 'Basic'],
            ['kiwitobes', 'France', 'yes', 23, 'Basic'], ['google', 'UK', 'no', 21, 'Premium'],
            ['(direct)', 'New Zealand', 'no', 12, 'None'],
            ['(direct)', 'UK', 'no', 21, 'Basic'],
            ['google', 'USA', 'no', 24, 'Premium'],
            ['slashdot', 'France', 'yes', 19, 'None'],
            ['digg', 'USA', 'no', 18, 'None'],
            ['google', 'UK', 'no', 18, 'None'],
            ['kiwitobes', 'UK', 'no', 19, 'None'],
            ['digg', 'New Zealand', 'yes', 12, 'Basic'],
            ['slashdot', 'UK', 'no', 21, 'None'],
            ['google', 'UK', 'yes', 18, 'Basic'],
            ['kiwitobes', 'France', 'yes', 19, 'Basic']]


class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def entropy(rows):
    """
    This function returns the entropy across the rows that have been supplied to it
    Entropy is the sum of -p(x)log(p(x)) across all the different sets
    :param rows:
    :return:
    """
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)

    # Now calculate the entropy
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def get_unique_counts(rows, column):
    return uniquecounts([[rows[i][column]] for i in range(len(rows))])


def uniquecounts(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows, column, value, greater_equal=0):
    # Make a function that tells us if a row is in
    # the first group (true) or the second group (false)
    split_function = None
    if (isinstance(value, int) or isinstance(value, float)) and greater_equal:
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def build_tree(rows, level, rank):
    if level == len(rank):
        return decisionnode(results=rows[0][len(rows[0]) - 1])
    set_count = get_unique_counts(rows, rank[level])
    root = None
    remaining_rows = rows
    for value in sorted(set_count):
        entry_rows, remaining_rows = divideset(remaining_rows, rank[level], value, 0)
        if root is None:
            root = decisionnode(rank[level], value, None, build_tree(entry_rows, level + 1, rank), None)
            prev_node = root
        else:
            prev_node.fb = decisionnode(rank[level], value, None, build_tree(entry_rows, level + 1, rank), None)
            prev_node = prev_node.fb
    return root
    pass


def buildtree(rows, scoreref=entropy):
    """
    This function accepts a dataset and generates a ranking based on the Dataset
    It calculates the Entropy and Information Gain for every column in the DataSet
    and ranks them. It then returns the rankings

    :param rows: The Dataset containing the data for whom the tree needs to be built

    :return: list. Returns a list containing the rank of each column, based on entropy and information gain
    """
    if len(rows) == 0:
        return decisionnode()

    # Caclculate the current score/entropy
    current_score = scoreref(rows)

    # Get the number of rows in the Data set
    number_of_rows = len(rows)
    # Get the number of columns in the Data set (This is assuming the dataset to not be jagged
    number_of_columns = len(rows[0])

    # Set the Information Gain for every column as 0
    information_gain = [0] * number_of_columns

    # Iterate over the number of columns
    # the list(map....)) transposes the list of lists
    for column_index, column in enumerate(list(map(list, zip(*rows)))):
        set_count = {x: column.count(x) for x in column}
        remaining_rows = rows

        # Set the Final Entropy, Best Entropy and the Best Value to 0
        final_entropy = 0
        best_entropy, best_value = 0, 0

        # For every 'value' in the set
        for value in set_count:
            entry_rows, remaining_rows = divideset(remaining_rows, column_index, value)

            # Calculate the final entropy
            final_entropy += (set_count[value] / number_of_rows) * entropy(entry_rows)

            # FOr numerical value, calculate it entropyin a differnet manner
            if isinstance(value, int) or isinstance(value, float):
                numerical_entropy = 0
                num_value, num_remain = divideset(rows, column_index, value, 1)
                numerical_entropy += (len(num_value) / number_of_rows) * entropy(num_value)
                numerical_entropy += ((number_of_rows - len(num_value)) / number_of_rows) * entropy(num_remain)

                if best_entropy == 0 or best_entropy > numerical_entropy:
                    best_entropy = numerical_entropy
                    best_value = num_value

        if isinstance(value, int) or isinstance(value, float):
            print(best_value, best_entropy)

        # Get the current information gain, that is, dataset entropy - set entropy
        current_info_gain = current_score - final_entropy

        # Set the column's info gain
        information_gain[column_index] = current_info_gain

        rank = [i for i in range(number_of_columns)]
        for i in range(len(rank) - 1):
            for j in range(i + 1, len(rank) - 1):
                if information_gain[i] < information_gain[j]:
                    temp = rank[j]
                    rank[j] = rank[i]
                    rank[i] = temp
        print(column_index, final_entropy)

    root = build_tree(rows, 0, rank[:len(rank) - 1])

    best_gain = 0
    best_criteria = None
    best_sets = None

    drawtree(root)

    pass


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def getwidth(tree):
    if tree == None or tree.results != None:
        return 1
    else:
        return getwidth(tree.tb) + getwidth(tree.fb) + 1


def getdepth(tree):
    if tree == None or tree.results != None:
        return 1
    else:
        return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


def drawnode(draw, tree, x, y):
    if tree == None:
        txt = 'untrained'
        draw.text((x - 20, y), txt, (0, 0, 0))
    elif tree.results == None:

        # Get the width of each branch
        w1 = getwidth(tree.tb) * 100
        w2 = getwidth(tree.fb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.tb, left + w1 / 2, y + 100)
        drawnode(draw, tree.fb, right - w2 / 2, y + 100)
    else:
        txt = tree.results
        draw.text((x - 20, y), txt, (0, 0, 0))


def main():
    # Get the data (assuming this to be a API call, that will fetch me data that I need)
    website_dataset = get_website_dataset()

    # Pass this data into the buildtree() function
    buildtree(website_dataset)


if __name__ == '__main__':
    main()
