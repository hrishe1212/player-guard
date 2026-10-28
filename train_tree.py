import csv
import math
import json
import pickle

# Pure Python Decision Tree Implementation (Cart Algorithm)
# Guarantees 100% portability, zero DLL dependencies, works on any Python version,
# and exactly implements max_depth=3 and max_leaf_nodes=None.

class Node:
    def __init__(self, depth=0):
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.value = None
        self.prob = 0.0
        self.samples = 0

def gini_impurity(y, weights):
    if not y:
        return 0.0
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    p1 = sum(w for yi, w in zip(y, weights) if yi == 1) / total_w
    p0 = 1.0 - p1
    return 1.0 - (p0**2 + p1**2)

class PureDecisionTreeClassifier:
    def __init__(self, max_depth=3, class_weight='balanced'):
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.root = None
        self.feature_names = []
        self.n_classes_ = 2

    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        n_samples = len(y)
        pos = sum(y)
        neg = n_samples - pos
        # Balanced weights
        w0 = n_samples / (2.0 * neg) if neg > 0 else 1.0
        w1 = n_samples / (2.0 * pos) if pos > 0 else 1.0
        weights = [w1 if yi == 1 else w0 for yi in y]

        self.root = self._build_tree(X, y, weights, depth=0)
        return self

    def _build_tree(self, X, y, weights, depth):
        node = Node(depth=depth)
        node.samples = len(y)
        total_w = sum(weights)
        w_pos = sum(w for yi, w in zip(y, weights) if yi == 1)
        node.prob = w_pos / total_w if total_w > 0 else 0.0
        node.value = 1 if node.prob >= 0.5 else 0

        # Check stopping criteria
        if depth >= self.max_depth or len(set(y)) <= 1 or len(y) < 4:
            node.is_leaf = True
            return node

        best_gain = -1.0
        best_feat = None
        best_thresh = None
        best_left_idx = None
        best_right_idx = None

        current_impurity = gini_impurity(y, weights)
        n_features = len(X[0])

        for f_idx in range(n_features):
            values = [row[f_idx] for row in X]
            sorted_vals = sorted(list(set(values)))
            if len(sorted_vals) <= 1:
                continue
            # Sample thresholds (percentiles for fast & robust split)
            step = max(1, len(sorted_vals) // 20)
            candidate_thresholds = [
                0.5 * (sorted_vals[i] + sorted_vals[i + 1])
                for i in range(0, len(sorted_vals) - 1, step)
            ]

            for thresh in candidate_thresholds:
                left_idx = [i for i, v in enumerate(values) if v <= thresh]
                right_idx = [i for i, v in enumerate(values) if v > thresh]

                if not left_idx or not right_idx:
                    continue

                left_y = [y[i] for i in left_idx]
                left_w = [weights[i] for i in left_idx]
                right_y = [y[i] for i in right_idx]
                right_w = [weights[i] for i in right_idx]

                w_l = sum(left_w)
                w_r = sum(right_w)

                gain = current_impurity - (w_l / total_w * gini_impurity(left_y, left_w) +
                                           w_r / total_w * gini_impurity(right_y, right_w))

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = thresh
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        if best_gain <= 1e-6 or best_feat is None:
            node.is_leaf = True
            return node

        node.feature = best_feat
        node.threshold = best_thresh
        node.left = self._build_tree(
            [X[i] for i in best_left_idx],
            [y[i] for i in best_left_idx],
            [weights[i] for i in best_left_idx],
            depth + 1
        )
        node.right = self._build_tree(
            [X[i] for i in best_right_idx],
            [y[i] for i in best_right_idx],
            [weights[i] for i in best_right_idx],
            depth + 1
        )
        return node

    def predict_one(self, x):
        curr = self.root
        while not curr.is_leaf:
            if x[curr.feature] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.value, curr.prob

    def predict(self, X):
        return [self.predict_one(x)[0] for x in X]

    def predict_proba(self, X):
        probs = []
        for x in X:
            _, p1 = self.predict_one(x)
            probs.append([1.0 - p1, p1])
        return probs

    def export_rules(self, node=None, prefix=""):
        if node is None:
            node = self.root
        lines = []
        if node.is_leaf:
            cls_name = "Elevated Risk (Class 1)" if node.value == 1 else "Lower Risk (Class 0)"
            lines.append(f"{prefix}--> Class: {cls_name} (Prob Class 1: {node.prob:.2%}, Samples: {node.samples})")
        else:
            feat_name = self.feature_names[node.feature]
            lines.append(f"{prefix}IF {feat_name} <= {node.threshold:.3f}:")
            lines.extend(self.export_rules(node.left, prefix + "    "))
            lines.append(f"{prefix}ELSE ({feat_name} > {node.threshold:.3f}):")
            lines.extend(self.export_rules(node.right, prefix + "    "))
        return lines

if __name__ == '__main__':
    print("Loading sports_multimodal_data.csv...")
    with open('sports_multimodal_data.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        features = header[:-1]
        X, y = [], []
        for row in reader:
            X.append([float(v) for v in row[:-1]])
            y.append(int(float(row[-1])))

    print(f"Training PureDecisionTreeClassifier with max_depth=3, max_leaves=unlimited...")
    dt = PureDecisionTreeClassifier(max_depth=3, class_weight='balanced')
    dt.fit(X, y, features)
    preds = dt.predict(X)

    tp = sum(1 for p, yt in zip(preds, y) if p == 1 and yt == 1)
    fp = sum(1 for p, yt in zip(preds, y) if p == 1 and yt == 0)
    tn = sum(1 for p, yt in zip(preds, y) if p == 0 and yt == 0)
    fn = sum(1 for p, yt in zip(preds, y) if p == 0 and yt == 1)

    acc = (tp + tn) / len(y)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"Accuracy: {acc:.2%}, Recall: {rec:.2%}, Precision: {prec:.2%}")
    print("\nDecision Tree Rules:")
    rules = dt.export_rules()
    for r in rules:
        print(r)

    # Save model to disk
    with open('trained_decision_tree.pkl', 'wb') as f:
        pickle.dump(dt, f)
    print("Saved trained model to trained_decision_tree.pkl successfully!")
