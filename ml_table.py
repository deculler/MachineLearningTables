from datascience import Table
from numbers import Number
import numpy as np
import matplotlib.pyplot as plots
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import linear_model, neighbors, discriminant_analysis

from matplotlib import colors
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plots.cm.register_cmap(cmap=cmap)

# Regression objects that provide a simple functional abstraction
# retain scikit_learn functionality
# and the R perspective

class Regression():
    """Container for KNN clasifiers object."""
    def __init__(self, obj, model, 
                 source_table=None, output_label=None, input_labels=None):
        self.obj = obj
        self.model = model
        self.source_table = source_table
        self.input_labels = input_labels
        self.output_label = output_label

class Linear(Regression):
    """Container for Linear regression object, ordinary and Ridge."""
    def __init__(self, obj, params, model, 
                 source_table=None, output_label=None, input_labels=None):
        Regression.__init__(self, obj, model, source_table, output_label, input_labels)
        self.params = params

    def summary(self):
        b0, bs = self.params
        sum_tbl = Table().with_columns([("Param", ['Intercept']+self.input_labels),
                                        ("Coefficient", [b0]+list(bs)),
                                        ])
        sum_tbl['Std Error'] = self.source_table.SE_params(self.output_label, (b0, bs))
        sum_tbl['t-statistic'] = sum_tbl['Coefficient'] / sum_tbl['Std Error']
        sum_tbl['95% CI'] = [(b-2*se, b+2*se) for b,se in zip(sum_tbl['Coefficient'], sum_tbl['Std Error'])]
        sum_tbl['99% CI'] = [(b-3*se, b+3*se) for b,se in zip(sum_tbl['Coefficient'], sum_tbl['Std Error'])]
        return sum_tbl

class Logit(Regression):
    """Container for Logistic regression object."""
    def __init__(self, obj, params, model, 
                 source_table=None, output_label=None, input_labels=None):
        Regression.__init__(self, obj, model, source_table, output_label, input_labels)
        self.params = params

    def likelihood(self, *args):
        b0, bs = self.params
        e = np.exp(b0 + np.sum(bs*args))
        return  e/(1 + e)
    def summary(self):
        b0, bs = self.params
        sum_tbl = Table().with_columns([("Param", ['Intercept']+self.input_labels),
                                        ("Coeffient", [b0]+list(bs)),
                                        ])
        return sum_tbl

class Knn(Regression):
    """Container for KNN clasifiers object."""
    def __init__(self, obj, model, 
                 source_table=None, output_label=None, input_labels=None):
        Regression.__init__(self, obj, model, source_table, output_label, input_labels)

class LinearDA(Regression):
    """Container for Logistic regression object."""
    def __init__(self, obj, params, model, 
                 source_table=None, output_label=None, input_labels=None):
        Regression.__init__(self, obj, model, source_table, output_label, input_labels)
        self.params = params

######################################

class ML_Table(Table):
    """Table with ML operators defined"""

    def __init__(self, *args, **kwargs):
        Table.__init__(self, *args, **kwargs)
            
    @classmethod
    def from_table(cls, tbl):
        ml_tbl = ML_Table()
        for label in tbl.labels:
            ml_tbl[label] = tbl[label]
        return ml_tbl
    
    # Utilities

    def _input_labels(self, output_label, input_labels=None):
        if input_labels is None:
            return [lbl for lbl in self.labels if lbl != output_label]
        else:
            return self._as_labels(input_labels)

    # Column generators
    @classmethod
    def sequence(cls, label, n, low=0, high=1):
        """Generate a table is a labeled column containing an 
        arithmetic sequence from low to high of length n."""
        return ML_Table().with_column(label, np.arange(low, high, (high-low)/n))
    
    @classmethod
    def rnorm(cls, label, n, mean=0, sd=1, seed=None):
        """Generate a table is a labeled column containing a random normal sequence of length n."""
        if seed is not None:
            np.random.seed(seed)
        return ML_Table().with_column(label, np.random.normal(loc=mean, scale=sd, size=n))

    @classmethod
    def runiform(cls, label, n, lo=-1, hi=1, seed=None):
        """Generate a table with a labeled column containing a uniform random sequence of length n over [lo, hi)."""
        if seed is not None:
            np.random.seed(seed)
        return ML_Table().with_column(label, np.random.rand(n)*(hi-lo) + lo)
          
    # Descriptive Statistics

    def summary(self, ops=None):
        """Generate a table corresponding to the R summary operator."""
        def FirstQu(x):
            return np.percentile(x, 25)
        def ThirdQu(x):
            return np.percentile(x, 5)
        if ops is None:
            ops=[min, FirstQu, np.median, np.mean, ThirdQu, max]
        return self.stats(ops=ops)

    # Regression methods for data fitting - 1d special case

    def regression_1d_params(self, Y_label_or_column, x_label_or_column):
        """Return parameters of a linear model of f(x) = Y."""
        x_values = self._get_column(x_label_or_column)
        Y_values = self._get_column(Y_label_or_column)
        m, b = np.polyfit(x_values, Y_values, 1)
        return b, m
    
    def regression_1d(self, Y_label_or_column, x_label_or_column):
        """Return a function that is a linear model of f(x) = Y."""
        b, m = self.regression_1d_params(Y_label_or_column, x_label_or_column)
        return lambda x: m*x + b

    # Regression methods for data fitting

    def poly_params(self, Y_label, x_label, degree):
        """Return a function that is a polynomial  model of f(x) = Y."""
        return np.polyfit(self[x_label], self[Y_label], degree)

    def poly(self, Y_label, x_label, degree):
        """Return a function that is a polynomial  model of f(x) = Y."""
        coefs = self.poly_params(Y_label, x_label, degree)
        def model(x):
            psum = coefs[0]
            for c in coefs[1:]:
                psum = x*psum + c
            return psum
        return model

    def _regression_obj(self, method, output_label, input_labels=None, **kwargs):
        """Generic pattern of sklearn classifier and regression usage."""
        input_labels = self._input_labels(output_label, input_labels)
        input_tbl = self.select(input_labels)
        regressor = method(**kwargs)
        regressor.fit(input_tbl.rows, self[output_label])
        return regressor


    def linear_regression(self, output_label, input_labels=None, **kwargs):
        """Return a linear regression function trained on a ML_Table.

        For kwargs and documentation see 


        """
        input_labels = self._input_labels(output_label, input_labels)
        obj = self._regression_obj(linear_model.LinearRegression,
                                   output_label, input_labels, **kwargs)
        params = obj.intercept_, obj.coef_
        model =  lambda *args: obj.predict([args])[0]
        return Linear(obj, params, model, self, output_label, input_labels)

    def ridge_regression(self, output_label, input_labels=None, **kwargs):
        """Return a linear ridge regression function trained on a ML_Table.

        For kwargs and documentation see 
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

        class sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, 
        copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)[source]¶
        """
        input_labels = self._input_labels(output_label, input_labels)
        obj = self._regression_obj(linear_model.Ridge,
                                   output_label, input_labels, **kwargs)
        params = obj.intercept_, obj.coef_
        model =  lambda *args: obj.predict([args])[0]
        return Linear(obj, params, model, self, output_label, input_labels)

    def knn_regression(self, output_label, input_labels=None, **kwargs):
        """Return a knn function trained on a ML_Table.

        For kwargs and documentation see 
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        5.25.4 sklearn.neighbors.KNeighborsRegressor

        class sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights=’uniform’, 
          algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’,
          metric_params=None, n_jobs=1, **kwargs)
        """
        input_labels = self._input_labels(output_label, input_labels)
        obj = self._regression_obj(neighbors.KNeighborsRegressor,
                                              output_label, input_labels, **kwargs)

        model =  lambda *args: obj.predict([args])[0]
        return Knn(obj, model, self, output_label, input_labels)

    def logit_regression(self, output_label, input_labels=None, **kwargs):
        """Return a logistic regression function trained on a ML_Table.

        For kwargs and documentation see 
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

        class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001,
          C=1.0, fit_intercept=True, intercept_scaling=1,
          class_weight=None,
          random_state=None, solver=’liblinear’,
          max_iter=100, multi_class=’ovr’, verbose=0,
          warm_start=False, n_jobs=1)

        """
        input_labels = self._input_labels(output_label, input_labels)
        logit_obj = self._regression_obj(linear_model.LogisticRegression,
                                                output_label, input_labels, **kwargs)
        logit_params = logit_obj.intercept_[0], logit_obj.coef_[0]
        logit_model =  lambda *args: logit_obj.predict([args])[0]
        return Logit(logit_obj, logit_params, logit_model, self, output_label, input_labels)

    def LDA(self, output_label, input_labels=None, **kwargs):
        """Return a linear discriminant analysis trained on a ML_Table.

        For kwargs and documentation see 
        http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

        class sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', 
        shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)

        This version is assuming 1 feature
        """
        input_labels = self._input_labels(output_label, input_labels)
        lda_obj = self._regression_obj(discriminant_analysis.LinearDiscriminantAnalysis,
                                                output_label, input_labels, **kwargs)
        lda_params = lda_obj.intercept_[0], lda_obj.coef_[0]
        lda_model =  lambda *args: lda_obj.predict([args])[0]
        return LinearDA(lda_obj, lda_params, lda_model, self, output_label, input_labels)

    # Statistics used in assessing models and understanding data

    def Cor_coef(self, x_column_or_label, y_column_or_label):
        """Computer the correlation coefficient between two columns."""
        x_values = self._get_column(x_column_or_label)
        y_values = self._get_column(y_column_or_label)
        x_res = x_values - np.mean(x_values)
        y_res = y_values - np.mean(y_values)
        return np.sum(x_res * y_res) / (np.sqrt(np.sum(x_res**2)) * np.sqrt(sum(y_res**2)))

    def Cor(self):
        """Create a correlation matrix of numeric columns as a table."""
        assert(self.num_rows > 0)
        num_labels = [lbl for lbl in self.labels if isinstance(self[lbl][0], Number)]
        tbl = self.select(num_labels)
        Cor_tbl = Table().with_column("Param", num_labels)
        for lbl in num_labels:
            Cor_tbl[lbl] = [self.Cor_coef(lbl, xlbl) for xlbl in num_labels] 
        return Cor_tbl

    def TSS(self, y_column_or_label):
        """Calulate the total sum of squares of observation column y."""
        y_values = self._get_column(y_column_or_label)
        y_mean = np.mean(y_values)
        return np.sum((y_values-y_mean)**2)

    def RSS(self, y_column_or_label, f_column_or_label):
        """Calulate the residual sum of square of observations y from estimate f."""
        y_values = self._get_column(y_column_or_label)
        f_values = self._get_column(f_column_or_label)
        return np.sum((y_values - f_values)**2)
    
    def MSE(self, y_column_or_label, f_column_or_label):
        """Calulate the mean squared error of a observations y from estimate f."""
        return self.RSS(y_column_or_label, f_column_or_label)/self.num_rows

    def RSE(self, y_column_or_label, f_column_or_label):
        """Computer the residual standard error of estimate f."""
        return np.sqrt(self.RSS(y_column_or_label, f_column_or_label)/(self.num_rows - 2))

    def R2(self, y_column_or_label, f_column_or_label):
        """Calulate the R^2 statistic of estimate of output y from f ."""
        return 1 - (self.RSS(y_column_or_label, f_column_or_label)/self.TSS(y_column_or_label))

    def F_stat(self, y_column_or_label, f_column_or_label, p):
        """Calulate the F-statistic of estimate f over p parameters ."""
        f_values = self._get_column(f_column_or_label)
        n = len(f_values)
        rss = self.RSS(y_column_or_label, f_values)
        tss = self.TSS(y_column_or_label)
        return ((tss-rss)/p) / (rss/(n - p - 1))

    def leverage_1d(self, x_column_or_label):
        """Calulate the 1d leverage statistic of an input column."""
        x_values = self._get_column(x_column_or_label)
        x_mean = np.mean(x_values)
        x_ss = np.sum((x_values - x_mean)**2)
        return ((x_values - x_mean)**2)/x_ss  + (1/len(x_values))

    # Category density

    def classification_error(self, y_column_or_label, f_column_or_label):
        y_values = self._get_column(y_column_or_label)
        f_values = self._get_column(f_column_or_label)
        correct = np.count_nonzero(y_values == f_values)
        return (len(y_values)-correct)/len(y_values)
        
    def density(self, output_label, input_label, bins=20, counts=False):
        """Generate a table containing the density of of inputs for each 
        key in a categorical output.
        """
        cat_counts = self.pivot_bin(output_label, input_label, bins=bins, normed=False)
        cat_counts.relabel('bin', input_label)
        if counts:
            return cat_counts
        totals = [np.sum(row[1:]) for row in cat_counts.rows]
        cat_density = cat_counts.select(input_label)
        for label in cat_counts.labels[1:]:
            cat_density[label] = cat_counts[label]/totals
        return cat_density

    # Common statistics from model functions - 1D special case

    def RSS_model_1d(self, y_column_or_label, model_fun, x_column_or_label):
        f_values = model_fun(self._get_column(x_column_or_label))
        return self.RSS(y_column_or_label, f_values)

    def R2_model_1d(self, y_column_or_label, model_fun, x_column_or_label):
        f_values = model_fun(self._get_column(x_column_or_label))
        return self.R2(y_column_or_label, f_values)

    def SE_1d_params(self, y_column_or_label, x_column_or_label, model = None):
        """Return the Standard Error of the parameters for a 1d regression."""
        x_values = self._get_column(x_column_or_label)
        x_mean = np.mean(x_values)
        x_dif_sq = np.sum((x_values - x_mean)**2)
        n = self.num_rows
        if model is None:
            model = self.regression_1d(y_column_or_label, x_values)
        sigma_squared = (self.RSS_model_1d(y_column_or_label, model, x_values))/(n-2)
        SE_b0_squared = sigma_squared*(1/n + (x_mean**2)/x_dif_sq) # constant term
        SE_b1_squared = sigma_squared/x_dif_sq # linear term
        return np.sqrt(SE_b0_squared), np.sqrt(SE_b1_squared)

    def lm_summary_1d(self, y_column_or_label, x_label):
        b0, b1 = self.regression_1d_params(y_column_or_label, x_label)
        r_model = lambda x: b0 + x*b1 
        SE_b0, SE_b1 = self.SE_1d_params(y_column_or_label, x_label, r_model)
        sum_tbl = Table().with_column('Param', ['intercept', x_label])
        sum_tbl['Coefficient'] = [b0, b1]
        sum_tbl['Std Error'] = (SE_b0, SE_b1)
        sum_tbl['t-statistic'] = np.array([b0, b1])/sum_tbl['Std Error']
        sum_tbl['95% CI'] = [(b0-2*SE_b0, b0+2*SE_b0), (b1-2*SE_b1, b1+2*SE_b1)]
        sum_tbl['99% CI'] = [(b0-3*SE_b0, b0+3*SE_b0), (b1-3*SE_b1, b1+3*SE_b1)]
        return sum_tbl


    # Common statistics from model functions - general case

    def f_values(self, output_label, model_fun, input_labels=None):
        input_labels = self._input_labels(output_label, input_labels)
        return [model_fun(*row) for row in self.select(input_labels).rows]        


    def classification_error_model(self, output_label, model_fun, input_labels=None):
        """Compute the residual sum of squares (RSS) for a model on a table."""
        f_values = self.f_values(output_label, model_fun, input_labels)
        return self.classification_error(output_label, f_values)

    def RSS_model(self, output_label, model_fun, input_labels=None):
        """Compute the residual sum of squares (RSS) for a model on a table."""
        f_values = self.f_values(output_label, model_fun, input_labels)
        return self.RSS(output_label, f_values)

    def R2_model(self, output_label, model_fun, input_labels=None):
        """Compute R^2 statistic for a model of a table.
        """
        f_values = self.f_values(output_label, model_fun, input_labels)
        return self.R2(output_label, f_values)

    def F_model(self, output_label, model_fun, input_labels=None):
        """Compute f-statistic for a model of a table.
        """
        p = len(self._input_labels(output_label)) if input_labels is None else 1
        f_values = self.f_values(output_label, model_fun, input_labels)
        return self.F_stat(output_label, f_values, p)

    def RSE_model(self, output_label, model_fun, input_labels=None):
        f_values = self.f_values(output_label, model_fun, input_labels)
        return self.RSE(output_label, f_values)

    def MSE_model(self, output_label, model_fun, input_labels=None):
        f_values = self.f_values(output_label, model_fun, input_labels)
        return self.MSE(output_label, f_values)

    def LOOCV_model(self, output_label, method, input_labels=None):
        """Computer the leave out one cross validation of a modeling method applied to
        a training table."""
        n = self.num_rows
        MSEs = [loo.MSE_model(output_label, loo.method(output_label, input_labels).model,
                              input_labels) for loo in [self.exclude(i) for i in range(n)]
                ]
        return np.sum(MSEs)/n
        
    
    def SE_params(self, output_label, params, input_labels=None):
        """Return the standard error of the parameters of a regression."""

        if input_labels is None:
            input_labels = self._input_labels(output_label)

        Y = self[output_label]  # Response vector
        n = len(Y)              # Number of points
        p = len(input_labels)   # Number of paraeters
        # Design matrix
        X = np.array([np.append([1], row) for row in self.select(input_labels).rows])
        b0, slopes = params
        b = np.append([b0], slopes)          # slope vector
        residual = np.dot(X, b) - Y        
        sigma2 = np.sum(residual**2)/(n-p-1)
        # standard error matrix
        std_err_matrix = sigma2*np.linalg.inv(np.dot(np.transpose(X), X))
        coef_std_err = [np.sqrt(std_err_matrix[i,i]) for i in range(len(std_err_matrix))]
        return coef_std_err

    def lm_summary(self, output_label):
        intercept, slopes = self.regression_params(output_label)
        mdl = ML_Table._make_model(intercept, slopes)
        input_labels = [lbl for lbl in self.labels if not lbl == output_label]        
        sum_tbl = Table().with_column('Param', ['Intercept'] + input_labels)
        sum_tbl['Coefficient'] = [intercept] + list(slopes)
        sum_tbl['Std Error'] = self.SE_params(output_label, (intercept, slopes))
        sum_tbl['t-statistic'] = sum_tbl['Coefficient'] / sum_tbl['Std Error']
        sum_tbl['95% CI'] = [(b-2*se, b+2*se) for b,se in zip(sum_tbl['Coefficient'], sum_tbl['Std Error'])]
        sum_tbl['99% CI'] = [(b-3*se, b+3*se) for b,se in zip(sum_tbl['Coefficient'], sum_tbl['Std Error'])]
        return sum_tbl

    def lm_fit(self, output_label, model_fun, x_column_or_label=None):
        if x_column_or_label is None:
            input_labels = [lbl for lbl in self.labels if not lbl == output_label]        
            f_values = [model_fun(*row) for row in self.select(input_labels).rows]
            p = len(input_labels)
        else:
            f_values = model_fun(self._get_column(x_column_or_label))
            p = 1
        fit_tbl = Table(["Quantity", "Value"])
        return fit_tbl.with_rows([("Residual standard error", self.RSE(output_label, f_values)),
                                  ("R^2", self.R2(output_label, f_values)),
                                  ("F-statistic", self.F_stat(output_label, f_values, p))])

    # Visualization

    def _plot_contour(f, x_lo, x_hi, y_lo, y_hi, n=20, **kwargs):        
        """Helper to form contour plot of a function over a 2D domain."""
        x_step = (x_hi - x_lo)/n
        y_step = (y_hi - y_lo)/n
        x_range = np.arange(x_lo, x_hi, x_step)
        y_range = np.arange(y_lo, y_hi, y_step)
        X, Y = np.meshgrid(x_range, y_range)
        Z = [[f(x,y) for x in x_range] for y in y_range]
        fig, ax = plots.subplots()
        CS = ax.contour(X, Y, Z, **kwargs)
        ax.clabel(CS, inline=2, fontsize=10)
        ax.grid(c='k', ls='-', alpha=0.3)
        return ax


    def RSS_contour(self, Y_column_or_label, x_column_or_label, scale=1,
                    sensitivity=0.1, n_grid=20, **kwargs):
        """Show contour of RSS around the regression point."""
        b0, b1 = self.regression_1d_params(Y_column_or_label, x_column_or_label)
        x_values = self._get_column(x_column_or_label)
        rss_fun = lambda b0,b1:self.RSS(Y_column_or_label, b0 + b1*x_values)*scale
        x_lo, x_hi = b0*(1-sensitivity), b0*(1+sensitivity)
        y_lo, y_hi = b1*(1-sensitivity), b1*(1+sensitivity)        
        ax = ML_Table._plot_contour(rss_fun, x_lo, x_hi, y_lo, y_hi, n = n_grid, **kwargs)
        ax.plot([b0], [b1], 'ro')
        return ax

    def RSS_contour2(self, output_label,
                     x_sensitivity=0.1, y_sensitivity = 0.1, scale=1,
                     n_grid=20, **kwargs):
        """Show contour of RSS around the 2-input regression point."""
        b0, coefs = self.linear_regression(output_label).params
        b1, b2 = coefs
        print(b0, b1, b2)
        x_lbl, y_lbl = self._input_labels(output_label)
        x_values, y_values = self[x_lbl], self[y_lbl]
        rss_fun = lambda b1,b2:self.RSS(output_label, b0 + b1*x_values + b2*y_values)*scale
        x_lo, x_hi = b1*(1-x_sensitivity), b1*(1+x_sensitivity)
        y_lo, y_hi = b2*(1-y_sensitivity), b2*(1+y_sensitivity)        
        ax = ML_Table._plot_contour(rss_fun, x_lo, x_hi, y_lo, y_hi, n = n_grid, **kwargs)
        ax.plot([b1], [b2], 'ro')
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)        
        return ax

    def _plot_wireframe(f, x_lo, x_hi, y_lo, y_hi, n=20, rstride=1, cstride=1, 
                        x_label=None, y_label=None, z_label=None):
        x_step = (x_hi - x_lo)/n
        y_step = (y_hi - y_lo)/n
        x_range = np.arange(x_lo, x_hi, x_step)
        y_range = np.arange(y_lo, y_hi, y_step)
        X, Y = np.meshgrid(x_range, y_range)
        Z = [[f(x,y) for x in x_range] for y in y_range]
        fig = plots.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, rstride=rstride, cstride=cstride, linewidth=1, color='b')
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_xlabel(y_label)
        if z_label is not None:
            ax.set_xlabel(z_label)
        return ax

    def RSS_wireframe(self, Y_column_or_label, x_column_or_label, 
                      sensitivity=0.1, n_grid=20):
        """Show wireframe of RSS surface around the regression point."""
        b0, b1 = self.regression_1d_params(Y_column_or_label, x_column_or_label)
        x_values = self._get_column(x_column_or_label)
        rss_fun = lambda b0,b1:self.RSS(Y_column_or_label, b0 + b1*x_values)
        x_lo, x_hi = b0*(1-sensitivity), b0*(1+sensitivity)
        y_lo, y_hi = b1*(1-sensitivity), b1*(1+sensitivity)        
        ax = ML_Table._plot_wireframe(rss_fun, x_lo, x_hi, y_lo, y_hi, n=n_grid)
        ax.scatter([b0], [b1], [rss_fun(b0, b1)], c='r')
        return ax


    def plot_fit_1d(self, y_label, x_label, model_fun, n_mesh=50, xmin=None, xmax=None,
                    width=6, height=4, connect=True, **kwargs):
        """Visualize the error in f(x) = y + error."""
        fig, ax = plots.subplots(figsize=(width, height))
        ax.scatter(self[x_label], self[y_label])
        f_tbl = self.select([x_label, y_label]).sort(x_label, descending=False)
        if model_fun is not None:
            if xmin is None:
                xmin = min(self[x_label])
            if xmax is None:
                xmax = max(self[x_label])
            xstep = (xmax-xmin)/n_mesh
            xv = np.arange(xmin, xmax + xstep, xstep)
            fun_x = [model_fun(x) for x in xv]
            ax.plot(xv, fun_x, **kwargs)
            if connect:
                for i in range(f_tbl.num_rows):
                    ax.plot([f_tbl[x_label][i], f_tbl[x_label][i]], 
                            [model_fun(f_tbl[x_label][i]), f_tbl[y_label][i] ], 'r-')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def plot_fit_2d(self, z_label, x_label, y_label, model_fun=None, n_mesh=50, 
                    xmin=None, xmax=None, ymin=None, ymax=None,
                    connect=True,
                    rstride=5, cstride=5, width=6, height=4, **kwargs):
        fig = plots.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection='3d')
        if model_fun is not None:
            if xmin is None:
                xmin = min(self[x_label])
            if xmax is None:
                xmax = max(self[x_label])
            if ymin is None:
                ymin = min(self[y_label])
            if ymax is None:
                ymax = max(self[y_label])
            xstep = (xmax-xmin)/n_mesh
            ystep = (ymax-ymin)/n_mesh
            xv = np.arange(xmin, xmax + xstep, xstep)
            yv = np.arange(ymin, ymax + ystep, ystep)
            X, Y = np.meshgrid(xv, yv)
            Z = [[model_fun(x,y) for x in xv] for y in yv]
            ax.plot_surface(X, Y, Z, rstride=5, cstride=5, linewidth=1, cmap=cm.coolwarm)
            ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=1, color='b', **kwargs)
            if connect:
                for (x, y, z) in zip(self[x_label], self[y_label], self[z_label]):
                    mz = model_fun(x,y)
                    ax.plot([x,x], [y,y], [z,mz], color='black')

        ax.scatter(self[x_label], self[y_label], self[z_label], c='r', marker='o')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        return ax

    def plot_fit(self, f_label, model_fun, width=6, height=4, **kwargs):
        """Visualize the goodness of fit of a model."""
        labels = [lbl for lbl in self.labels if not lbl == f_label]
        assert len(labels) <= 2, "Too many dimensions to plot"
        if len(labels) == 1:
            return self.plot_fit_1d(f_label, labels[0], model_fun, **kwargs)
        else:
            return self.plot_fit_2d(f_label, labels[0], labels[1], model_fun, 
                                    width=width, height=height, **kwargs)
                    

    def _plot_color(f, x_lo, x_hi, y_lo, y_hi, n=20, **kwargs):        
        """Helper to form colormap of a function over a 2D domain."""
        x_step = (x_hi - x_lo)/n
        y_step = (y_hi - y_lo)/n
        x_range = np.arange(x_lo, x_hi, x_step)
        y_range = np.arange(y_lo, y_hi, y_step)
        X, Y = np.meshgrid(x_range, y_range)
        Z = [[f(x,y) for x in x_range] for y in y_range]
        fig, ax = plots.subplots()
        ax.pcolormesh(X, Y, Z, cmap='red_blue_classes',
                      norm=colors.Normalize(0., 1.))
        CS = ax.contour(X, Y, Z, [0.5], **kwargs)
        ax.grid(c='k', ls='-', alpha=0.3)
        return ax

    def plot_cut_2d(self, cat_label, x_label, y_label, model_fun=None, n_grid=50, 
                    xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
        if xmin is None:
            xmin = min(self[x_label])
        if xmax is None:
            xmax = max(self[x_label])
        if ymin is None:
            ymin = min(self[y_label])
        if ymax is None:
            ymax = max(self[y_label])
        ax = ML_Table._plot_color(model_fun, xmin, xmax, ymin, ymax, n_grid, **kwargs)

        categories = np.unique(self[cat_label])
        colors = plots.cm.nipy_spectral(np.linspace(0, 1, len(categories)))
        for cat, color in zip(categories, colors):
            ax.scatter(self.where(cat_label, cat)[x_label],
                       self.where(cat_label, cat)[y_label], color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(categories, loc=2, bbox_to_anchor=(1.05, 1))
        return ax


# Cross validation
    
