def possible_thresholds(self, node, feature):
        """Calculates midpoints between unique values for thresholding."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Computes Gini impurity for a single feature efficiently via arrays."""
        sub_X = self.explanatory[node.sub_population, feature]
        sub_Y = self.target[node.sub_population]
        thresholds = self.possible_thresholds(node, feature)
        classes = np.unique(self.target[node.sub_population])

        n = sub_X.size
        t = thresholds.size
        c = classes.size
        
        if t == 0:
            return 0.0, 1.0 

        X_exp = sub_X.reshape(n, 1, 1)
        thresh_exp = thresholds.reshape(1, t, 1)
        Y_exp = sub_Y.reshape(n, 1, 1)
        C_exp = classes.reshape(1, 1, c)

        Left_F = (X_exp > thresh_exp) & (Y_exp == C_exp)
        Right_F = (X_exp <= thresh_exp) & (Y_exp == C_exp)

        L_pop = np.sum(X_exp > thresh_exp, axis=0)
        R_pop = np.sum(X_exp <= thresh_exp, axis=0)

        L_pop_s = np.where(L_pop == 0, 1.0, L_pop).astype(float)
        R_pop_s = np.where(R_pop == 0, 1.0, R_pop).astype(float)

        L_gini = 1.0 - np.sum((np.sum(Left_F, axis=0) / L_pop_s) ** 2, axis=1)
        R_gini = 1.0 - np.sum((np.sum(Right_F, axis=0) / R_pop_s) ** 2, axis=1)

        avg_gini = (L_pop.flatten() / n) * L_gini + (R_pop.flatten() / n) * R_gini
        best_i = np.argmin(avg_gini)
        
        return thresholds[best_i], avg_gini[best_i]

    def Gini_split_criterion(self, node):
        """Evaluates all features to find the optimal Gini split."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i) 
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
