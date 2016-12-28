from datascience import Table
import numpy as np
import matplotlib.pyplot as plots
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import linear_model

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
          
    def summary(self, ops=None):
        """Generate a table corresponding to the R summary operator."""
        def FirstQu(x):
            return np.percentile(x, 25)
        def ThirdQu(x):
            return np.percentile(x, 5)
        if ops is None:
            ops=[min, FirstQu, np.median, np.mean, ThirdQu, max]
        return self.stats(ops=ops)

    # Common statistical machine learning operators
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

    def regression_params(self, output_label):
        """Form a model of a table using linear regression and return as a function."""
        input_labels = [lbl for lbl in self.labels if not lbl == output_label]
        reg = linear_model.LinearRegression()
        input_vectors = [self[lbl] for lbl in input_labels]
        reg.fit(np.transpose(input_vectors), self[output_label])
        return  reg.intercept_, reg.coef_

    def ridge_params(self, output_label, **kwargs):
        """Form a model of a table using linear regression and return as a function."""
        input_labels = [lbl for lbl in self.labels if not lbl == output_label]
        reg = linear_model.Ridge(**kwargs)
        input_vectors = [self[lbl] for lbl in input_labels]
        reg.fit(np.transpose(input_vectors), self[output_label])
        return reg.intercept_, reg.coef_

    def _make_model(intercept, slopes):
        def _reg_fun(*args):
            psum = intercept
            for p,v in zip(slopes, args):
                psum += p*v
            return psum
        return _reg_fun

    def regression(self, output_label, method=None, **kwargs):
        """Make a function as model over input values using regression."""
        if method is None:
            method = self.regression_params
        b0, coefs = method(output_label, **kwargs)
        return ML_Table._make_model(b0, coefs)

    def ridge(self, output_label,  **kwargs):
        return self.regression(output_label, method=self.ridge_params, **kwargs)

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

    def Cor_coef(self, x_column_or_label, y_column_or_label):
        x_values = self._get_column(x_column_or_label)
        y_values = self._get_column(y_column_or_label)
        x_res = x_values - np.mean(x_values)
        y_res = y_values - np.mean(y_values)
        return np.sum(x_res * y_res) / (np.sqrt(np.sum(x_res**2)) * np.sqrt(sum(y_res**2)))

    def Cor(self):
        """Create a correlation matrix as a table."""
        Cor_tbl = Table().with_column("Param", self.labels)
        for lbl in self.labels:
            Cor_tbl[lbl] = [self.Cor_coef(lbl, xlbl) for xlbl in self.labels]
        return Cor_tbl

    def TSS(self, y_column_or_label):
        """Calulate the total sum of squares of observations y."""
        y_values = self._get_column(y_column_or_label)
        y_mean = np.mean(y_values)
        return np.sum((y_values-y_mean)**2)

    def RSS(self, y_column_or_label, f_column_or_label):
        """Calulate the sum of square error of a observations y from estimate f."""
        y_values = self._get_column(y_column_or_label)
        f_values = self._get_column(f_column_or_label)
        return np.sum((y_values - f_values)**2)

    def R2(self, y_column_or_label, f_column_or_label):
        """Calulate the R^2 statistic of estimate f ."""
        return 1 - (self.RSS(y_column_or_label, f_column_or_label)/self.TSS(y_column_or_label))

    def F_stat(self, y_column_or_label, f_column_or_label, p):
        """Calulate the F-statistic of estimate f over p parameters ."""
        f_values = self._get_column(f_column_or_label)
        n = len(f_values)
        rss = self.RSS(y_column_or_label, f_values)
        tss = self.TSS(y_column_or_label)
        return ((tss-rss)/p) / (rss/(n - p - 1))

    def RSE(self, y_column_or_label, f_column_or_label):
        return np.sqrt(self.RSS(y_column_or_label, f_column_or_label)/(self.num_rows - 2))
    
    def MSE(self, y_column_or_label, f_column_or_label):
        """Calulate the mean square error of a observations y from estimate f."""
        return self.RSS(y_column_or_label, f_column_or_label)/self.num_rows
    
    def RSS_model_1d(self, y_column_or_label, model_fun, x_column_or_label):
        f_values = model_fun(self._get_column(x_column_or_label))
        return self.RSS(y_column_or_label, f_values)

    def RSS_model(self, output_label, model_fun, x_column_or_label=None):
        """Compute the residual sum of squares (RSS) for a model on a table.
        """
        if x_column_or_label is None:
            input_labels = [lbl for lbl in self.labels if not lbl == output_label]        
            f_values = [model_fun(*row) for row in self.select(input_labels).rows]
        else:
            f_values = model_fun(self._get_column(x_column_or_label))
        return self.RSS(output_label, f_values)

    def R2_model_1d(self, y_column_or_label, model_fun, x_column_or_label):
        f_values = model_fun(self._get_column(x_column_or_label))
        return self.R2(y_column_or_label, f_values)

    def R2_model(self, output_label, model_fun, x_column_or_label=None):
        """Compute R^2 statistic for a model of a table.

        If x_column_or_label is specified, it is treated as a 1d input.
        Otherwise, all columns besides output_label are treated as inputs.
        """
        if x_column_or_label is None:
            input_labels = [lbl for lbl in self.labels if not lbl == output_label]        
            f_values = [model_fun(*row) for row in self.select(input_labels).rows]
        else:
            f_values = model_fun(self._get_column(x_column_or_label))
        return self.R2(output_label, f_values)

    def F_model(self, output_label, model_fun, x_column_or_label=None):
        """Compute f-statistic for a model of a table.
        """
        if x_column_or_label is None:
            input_labels = [lbl for lbl in self.labels if not lbl == output_label]        
            f_values = [model_fun(*row) for row in self.select(input_labels).rows]
            p = len(input_labels)
        else:
            f_values = model_fun(self._get_column(x_column_or_label))
            p = 1
        return self.F_stat(output_label, f_values, p)

    def RSE_model(self, y_column_or_label, model_fun, x_column_or_label):
        f_values = model_fun(self._get_column(x_column_or_label))
        return self.RSE(y_column_or_label, f_values)

    def MSE_model(self, y_column_or_label, model_fun, x_column_or_label):
        return self.RSS_model(y_column_or_label, model_fun, x_column_or_label)/self.num_rows
    
    def SE_1d_params(self, y_column_or_label, x_column_or_label, model = None):
        """Return the Standard Error of the parameters for a 1d regression."""
        x_values = self._get_column(x_column_or_label)
        x_mean = np.mean(x_values)
        x_dif_sq = np.sum((x_values - x_mean)**2)
        n = self.num_rows
        if model is None:
            model = self.regression_1d(y_column_or_label, x_values)
        sigma_squared = (self.RSS_model(y_column_or_label, model, x_values))/(n-2)
        SE_b0_squared = sigma_squared*(1/n + (x_mean**2)/x_dif_sq) # constant term
        SE_b1_squared = sigma_squared/x_dif_sq # linear term
        return np.sqrt(SE_b0_squared), np.sqrt(SE_b1_squared)

    def SE_params(self, output_label, params = None):
        """Return the standard error of the parameters of a regression."""
        if params is None:
            params = self.regression_params(output_label)
        input_labels = [lbl for lbl in self.labels if not lbl == output_label]
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

    def _plot_contour(f, x_lo, x_hi, y_lo, y_hi, n=20, levels=10):        
        """Helper to form contour plot of a function over a 2D domain."""
        x_step = (x_hi - x_lo)/n
        y_step = (y_hi - y_lo)/n
        x_range = np.arange(x_lo, x_hi, x_step)
        y_range = np.arange(y_lo, y_hi, y_step)
        X, Y = np.meshgrid(x_range, y_range)
        Z = [[f(x,y) for x in x_range] for y in y_range]
        fig, ax = plots.subplots()
        CS = ax.contour(X, Y, Z, levels)
        ax.clabel(CS, inline=2, fontsize=10)
        ax.grid(c='k', ls='-', alpha=0.3)
        return ax

    def RSS_contour(self, Y_column_or_label, x_column_or_label, 
                    sensitivity=0.1, n_grid=20, levels=10):
        """Show contour of RSS around the regression point."""
        b0, b1 = self.regression_1d_params(Y_column_or_label, x_column_or_label)
        x_values = self._get_column(x_column_or_label)
        rss_fun = lambda b0,b1:self.RSS(Y_column_or_label, b0 + b1*x_values)
        x_lo, x_hi = b0*(1-sensitivity), b0*(1+sensitivity)
        y_lo, y_hi = b1*(1-sensitivity), b1*(1+sensitivity)        
        ax = ML_Table._plot_contour(rss_fun, x_lo, x_hi, y_lo, y_hi, n = n_grid, levels=levels)
        ax.plot([b0], [b1], 'ro')
        return ax

    def _plot_wireframe(f, x_lo, x_hi, y_lo, y_hi, n=20, rstride=1, cstride=1):
        x_step = (x_hi - x_lo)/n
        y_step = (y_hi - y_lo)/n
        x_range = np.arange(x_lo, x_hi, x_step)
        y_range = np.arange(y_lo, y_hi, y_step)
        X, Y = np.meshgrid(x_range, y_range)
        Z = [[f(x,y) for x in x_range] for y in y_range]
        fig = plots.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, rstride=rstride, cstride=cstride, linewidth=1, color='b')
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


    def boxplot(self, column_for_xcats=None, select=None, height=4, width=6,  **vargs):
        """Plot box-plots for the columns in a table.

        If no column for categories is specified, 
        a boxplot is produced for each column (or for the columns designated
        by `select`) labeled by the column name.
        If one is satisfied, a box plot is produced for each other column
        using a pivot on the categories.

        Every selected column must be numerical, other than the category column

        Args:

        Kwargs:
            select (column name or list): columns to include

            vargs: Additional arguments that get passed into `plt.plot`.
                See http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
                for additional arguments that can be passed into vargs.
        """
        options = self.default_options.copy()
        options.update(vargs)

        x_labels = self.labels
        if select is not None:
            x_labels = self._as_labels(select)

        if column_for_xcats is None:
            fig, ax = plots.subplots(figsize=(width,height))
            data = [self[lbl] for lbl in x_labels]
            ax.boxplot(data, labels=x_labels, **vargs)
            return ax
        else:
            grouped = self.select(x_labels).group(column_for_xcats, collect=lambda x:x)
            x_labels = [lbl for lbl in x_labels if lbl != column_for_xcats]
            fig, axes = plots.subplots(len(x_labels), 1, figsize=(width, height))
            if len(x_labels) == 1:
                axes = [axes]
            for (lbl,axis) in zip(x_labels, axes):
                axis.boxplot(grouped[lbl], labels=grouped[column_for_xcats])
                axis.set_ylabel(lbl)
            if len(x_labels) == 1:
                return axes[0]
            else:
                return axes

    def plot_fit_1d(self, y_label, x_label, model_fun, **kwargs):
        """Visualize the error in f(x) = y + error."""
        fig, ax = plots.subplots()
        ax.scatter(self[x_label], self[y_label])
        f_tbl = self.select([x_label, y_label]).sort(x_label, descending=False)
        fun_x = f_tbl.apply(model_fun, x_label)
        ax.plot(f_tbl[x_label], fun_x, **kwargs)
        for i in range(f_tbl.num_rows):
            ax.plot([f_tbl[x_label][i], f_tbl[x_label][i]], 
                    [fun_x[i], f_tbl[y_label][i] ], 'r-')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def plot_fit_2d(self, z_label, x_label, y_label, model_fun=None, n_mesh=50, 
                    xmin=None, xmax=None, ymin=None, ymax=None,
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
            Z = model_fun(X, Y)
            ax.plot_surface(X, Y, Z, rstride=5, cstride=5, linewidth=1, cmap=cm.coolwarm)
            ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=1, color='b', **kwargs)
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
                    

    
