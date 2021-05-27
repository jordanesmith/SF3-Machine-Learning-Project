"""
fork from python-rl and pybrain for visualization
"""
#import numpy as np
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt

# If theta  has gone past our conceptual limits of [-pi,pi]
# map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
     
def remap_angle(theta):
    return _remap_angle(theta)
    
def _remap_angle(theta):
    while theta < -np.pi:
        theta += 2. * np.pi
    while theta > np.pi:
        theta -= 2. * np.pi
    return theta
    

## loss function given a state vector. the elements of the state vector are
## [cart location, cart velocity, pole angle, pole angular velocity]
def _loss(state):
    sig = 0.5
    return 1-np.exp(-np.dot(state,state)/(2.0 * sig**2))

def loss(state):
    return _loss(state)

class CartPole:
    """Cart Pole environment. This implementation allows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """

    def __init__(self, visual=False):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi    # angle is defined to be zero when the pole is upright, pi when hanging vertically down
        self.pole_velocity = 0.0
        self.visual = visual

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        #   or they vary by some scalar from the other poles)
        self.pole_length = 0.5 
        self.pole_mass = 0.5 

        self.mu_c = 0.001 #   # friction coefficient of the cart
        self.mu_p = 0.001 # # friction coefficient of the pole
        self.sim_steps = 50         # number of Euler integration steps to perform in one go
        self.delta_time = 0.2        # time step of the Euler integrator
        self.max_force = 20.
        self.gravity = 9.8
        self.cart_mass = 0.5

        # for plotting
        self.cartwidth = 1.0
        self.cartheight = 0.2

        if self.visual:
            self.drawPlot()

    def setState(self, state):
        self.cart_location = state[0]
        self.cart_velocity = state[1]
        self.pole_angle = state[2]
        self.pole_velocity = state[3]
            
    def getState(self):
        return np.array([self.cart_location,self.cart_velocity,self.pole_angle,self.pole_velocity])

    # reset the state vector to the initial state (down-hanging pole)
    def reset(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi
        self.pole_velocity = 0.0

    # This is where the equations of motion are implemented
    def performAction(self, action = 0.0):
        # prevent the force from being too large
        force = self.max_force * np.tanh(action/self.max_force)

        # integrate forward the equations of motion using the Euler method
        for step in range(self.sim_steps):
            s = np.sin(self.pole_angle)
            c = np.cos(self.pole_angle)
            m = 4.0*(self.cart_mass+self.pole_mass)-3.0*self.pole_mass*(c**2)
            
            cart_accel = (2.0*(self.pole_length*self.pole_mass*(self.pole_velocity**2)*s+force-self.mu_c*self.cart_velocity)\
                -3.0*self.pole_mass*self.gravity*c*s )/m
            
            pole_accel = (-3.0*c*(self.pole_length/2.0*self.pole_mass*(self.pole_velocity**2)*s + force-self.mu_c*self.cart_velocity)+\
                6.0*(self.cart_mass+self.pole_mass)/(self.pole_mass*self.pole_length)*\
                (self.pole_mass*self.gravity*s - 2.0/self.pole_length*self.mu_p*self.pole_velocity) \
                )/m

            # Update state variables
            dt = (self.delta_time / float(self.sim_steps))
            # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
            self.cart_velocity += dt * cart_accel
            self.pole_velocity += dt * pole_accel
            self.pole_angle    += dt * self.pole_velocity
            self.cart_location += dt * self.cart_velocity

        if self.visual:
            self._render()

    # remapping as a member function
    def remap_angle(self):
        self.pole_angle = _remap_angle(self.pole_angle)
    
    # the loss function that the policy will try to optimise (lower) as a member function
    def loss(self):
        return _loss(self.getState())
    
    def terminate(self):
        """Indicates whether or not the episode should terminate.

        Returns:
            A boolean, true indicating the end of an episode and false indicating the episode should continue.
            False is returned if either the cart location or
            the pole angle is beyond the allowed range.
        """
        return np.abs(self.cart_location) > self.state_range[0, 1] or \
               (np.abs(self.pole_angle) > self.state_range[2, 1]).any()

   # the following are graphics routines
    def drawPlot(self):
        ion()
        self.fig = plt.figure()
        # draw cart
        self.axes = self.fig.add_subplot(111, aspect='equal')
        self.box = Rectangle(xy=(self.cart_location - self.cartwidth / 2.0, -self.cartheight), 
                             width=self.cartwidth, height=self.cartheight)
        self.axes.add_artist(self.box)
        self.box.set_clip_box(self.axes.bbox)

        # draw pole
        self.pole = Line2D([self.cart_location, self.cart_location + np.sin(self.pole_angle)], 
                           [0, np.cos(self.pole_angle)], linewidth=3, color='black')
        self.axes.add_artist(self.pole)
        self.pole.set_clip_box(self.axes.bbox)

        # set axes limits
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-0.5, 2)
        


    def _render(self):
        self.box.set_x(self.cart_location - self.cartwidth / 2.0)
        self.pole.set_xdata([self.cart_location, self.cart_location + np.sin(self.pole_angle)])
        self.pole.set_ydata([0, np.cos(self.pole_angle)])
        self.fig.show()
        
        plt.pause(0.2)



def move_cart(initial_x, steps=10, visual=False, display_plots=True, remap_angle=False):
    """
    
    Parameters
    ----------
    initial_x : list-like
        [cart_location, cart_velocity, pole_angle, pole_velocity, action]
    steps: int
        number of steps taken
    visual: bool
        whether to show image of cart state
    display_plots: bool
        
    remap_angle: bool
        whether to remap angle to between pi -pi
        
    Returns
    -------
    list 
        x_history of state at discrete intervals
    """
    assert steps != 4, "Sorry, I don't like 4 steps"
    
    cp = CartPole(visual=visual)
    cp.cart_location, cp.cart_velocity, cp.pole_angle, cp.pole_velocity, action = initial_x
    
    for step in range(steps):
        if visual: cp.drawPlot()
        cp.performAction(action=action)
        if remap_angle: cp.remap_angle()
        try: 
            x_history = np.vstack((x_history, np.array([cp.cart_location, cp.cart_velocity, cp.pole_angle, cp.pole_velocity, action])))
        except:
            x_history = np.array((initial_x, [cp.cart_location, cp.cart_velocity, cp.pole_angle, cp.pole_velocity, action]))
        
    t = range(x_history.shape[0]) if steps > 1 else 1
    
    if display_plots and steps > 1:
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        
        axs[0].plot(t, [x[0] for x in x_history], label='cart_location')
        axs[0].plot(t, [x[1] for x in x_history], label='cart_velocity')
        axs[0].plot(t, [x[2] for x in x_history], label='pole_angle')
        axs[0].plot(t, [x[3] for x in x_history], label='pole_velocity')
        axs[0].legend()
        
        axs[1].plot([x[0] for x in x_history], [x[1] for x in x_history])
        axs[1].set_xlabel('cart_location')
        axs[1].set_ylabel('cart_velocity')
        
        axs[2].scatter([x[2] for x in x_history], [x[3] for x in x_history])
        axs[2].set_xlabel('pole_angle')
        axs[2].set_ylabel('pole_velocity')
        
        fig.suptitle('action: {}'.format(action))
        fig.tight_layout()
    
    elif display_plots and steps == 1: print("You're trying to plot over {} steps, which is not plottable, pick a number greater than 1".format(steps))
        
    if len(x_history) != 5: return x_history[-1]
    else: return x_history

def generate_data(n, steps=1, remap_angle=False):
    for i in range(n):
        try:
            x_ = np.array([np.random.normal(), np.random.uniform(-10, 10), np.random.uniform(-np.pi,np.pi), np.random.uniform(-15,15), np.random.uniform(-20,20)])
            y_ = np.array(move_cart(x_, steps=steps, display_plots=False, remap_angle=remap_angle)) - x_
            x = np.vstack((x, x_))
            y = np.vstack((y, y_))

        except:
            x = np.array([np.random.normal(), np.random.uniform(-10, 10), np.random.uniform(-np.pi,np.pi), np.random.uniform(-15,15), np.random.uniform(-20,20)])
            y = np.array(move_cart(x, steps=steps, display_plots=False, remap_angle=remap_angle)) - x
            
    return x,y

def plot_y_contour_as_difference_in_x(initial_x, index_pair, range_x_pair, index_to_variable, dynamics='actual', model=None, **kwargs):
    '''
    function for plotting y contours when y is modelled 
    as X(T) - X(0) and 2 variables are scanned across 
    
    Parameters
    ----------
    index_pair : list-like of int
        Which index pair of X (or variables) to scan over
    range_x_pair : list-like of list-like
        Scan range of both variables
    '''
   
    index_1, index_2 = index_pair
    range_1, range_2 = range_x_pair
    
    x_0_grid = np.zeros((len(range_1),len(range_2),5))
    x_t_grid = np.zeros((len(range_1),len(range_2),5))
    
    for i,value_1 in enumerate(range_1):
        for j, value_2 in enumerate(range_2):
            x = initial_x.copy()
            x[index_1] = value_1
            x[index_2] = value_2
            x_0_grid[i,j] = x
            if dynamics == 'actual': x_t_grid[i,j] = np.array(move_cart(x, steps=1, display_plots=False, remap_angle=False))
            elif dynamics == 'predicted':
                assert model, 'no model given'
                x_t_grid[i,j] = model(x, kwargs['alpha'], kwargs['X_i_vals'], kwargs['sigma']) # TODO make this model.predict()
    y_grid = x_t_grid - x_0_grid
    y_grid = np.moveaxis(y_grid, -1, 0)   
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
    if index_pair == [2,3]:
        vmin = None
        vmax = None
    else:
        vmin = y_grid.min()
        vmax = y_grid.max()
    axs[0,0].contourf(range_1, range_2, y_grid[0].T, vmin=vmin, vmax=vmax)
    axs[0,0].set_title('cart_location')
    axs[0,0].set_xlabel('{} initial value'.format(index_to_variable[index_1]))
    axs[0,0].set_ylabel('{} initial value'.format(index_to_variable[index_2]))    
    axs[0,1].contourf(range_1, range_2, y_grid[1].T, vmin=vmin, vmax=vmax)
    axs[0,1].set_title('cart_velocity')
    axs[0,1].set_xlabel('{} initial value'.format(index_to_variable[index_1]))
    axs[0,1].set_ylabel('{} initial value'.format(index_to_variable[index_2]))
    axs[1,0].contourf(range_1, range_2, y_grid[2].T, vmin=vmin, vmax=vmax)
    axs[1,0].set_title('pole_angle')
    axs[1,0].set_xlabel('{} initial value'.format(index_to_variable[index_1]))
    axs[1,0].set_ylabel('{} initial value'.format(index_to_variable[index_2]))
    axs[1,1].contourf(range_1, range_2, y_grid[3].T, vmin=vmin, vmax=vmax)
    axs[1,1].set_title('pole_velocity')
    axs[1,1].set_xlabel('{} initial value'.format(index_to_variable[index_1]))
    axs[1,1].set_ylabel('{} initial value'.format(index_to_variable[index_2]))
    
    if 4 not in index_pair: fig.suptitle('action: {}'.format(initial_x[-1]))
    fig.tight_layout()

def range_x_pair_finder(index_pair, x_range_for_index):
    range_x_pair = []
    for index in index_pair:
        range_x_pair.append(x_range_for_index[index])
    return range_x_pair

def project_x_using_model(initial_x, model, steps, remap_angle=False, compound_predictions=False, **kwargs):
    
    cp = CartPole()
    cp.cart_location, cp.cart_velocity, cp.pole_angle, cp.pole_velocity, action = initial_x
    pred_ = None
    
    for step in range(steps):
        x_ = np.array([cp.cart_location, cp.cart_velocity, cp.pole_angle, cp.pole_velocity, action])
        cp.performAction(action)
        if remap_angle: cp.remap_angle()
        y_ = np.array([cp.cart_location, cp.cart_velocity, cp.pole_angle, cp.pole_velocity, action])
        if pred_ is not None: 
            if compound_predictions:
                pred_ = pred_ + model(pred_, kwargs['alpha'], kwargs['X_i_vals'], kwargs['sigma']) #TODO change to model.predict
            else:
                pred_ = x_ + model(x_, kwargs['alpha'], kwargs['X_i_vals'], kwargs['sigma']) #TODO change to model.predict
          
        try:
            prediction_history = np.vstack((prediction_history, pred_))
            y_history = np.vstack((y_history, y_))
        except:
            assert all(x_) == all(initial_x), '{}_______{}'.format(x_, initial_x)
            pred_ = x_ + model(x_, kwargs['alpha'], kwargs['X_i_vals'], kwargs['sigma'])
            prediction_history = np.vstack((x_, pred_))
            y_history = np.vstack((x_, y_))
        print('action in project_x_using_model step {} was {}'.format(step, action))
    
    return prediction_history, y_history

def plot_prediction_vs_actual_over_time(prediction_history, y_history, title=None):
    
    t = range(len(prediction_history))
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs[0,0].plot(t, [y[0] for y in y_history], label='actual values')
    axs[0,0].plot(t, [pred[0] for pred in prediction_history], label='predicted values')
    axs[0,0].set_ylabel('Y_cart_location')
    axs[0,0].set_xlabel('time_step')    
    axs[0,1].plot(t, [y[1] for y in y_history], label='actual values')
    axs[0,1].plot(t, [pred[1] for pred in prediction_history], label='predicted values')
    axs[0,1].set_ylabel('Y_cart_velocity')
    axs[0,1].set_xlabel('time_step')    
    axs[1,0].plot(t, [y[2] for y in y_history], label='actual values')
    axs[1,0].plot(t, [pred[2] for pred in prediction_history], label='predicted values')
    axs[1,0].set_ylabel('Y_pole_angle')
    axs[1,0].set_xlabel('time_step')
    axs[1,1].plot(t, [y[3] for y in y_history], label='actual values')
    axs[1,1].plot(t, [pred[3] for pred in prediction_history], label='predicted values')
    axs[1,1].set_ylabel('Y_pole_velocity')
    axs[1,1].set_xlabel('time_step')
    axs[0,1].legend(loc='upper right')
    if title: descriptive_title = title
    else: descriptive_title = ''

    fig.suptitle(descriptive_title + ' action: {}'.format(y_history[0][-1]))
    fig.tight_layout()
    


def plot_y_scans(initial_x, index_to_variable, x_range_for_index, model=None, remap_angle=False, **kwargs):
    '''
    function for plotting y values when y is modelled 
    as X(T) - X(0)
    
    Parameters
    ----------
    model : 
        linear regression model
    '''
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    for index in range(4):
        
        range_x = x_range_for_index[index]
        
        x = initial_x.copy()
        y_results = []
        x_0 = None
        x_t = None

        for i in range_x:
            x[index] = i
            x_t = np.array(move_cart(x, steps=1, display_plots=False, remap_angle=remap_angle))
            try:
                x_t_results = np.vstack((x_t_results, x_t))
                x_0 = np.vstack((x_0, x))
            except:
                x_t_results = x_t
                x_0 = x.copy()

        if model: 
            try: 
                predictions = model(x_0, kwargs['alpha'], kwargs['X_i_vals'], kwargs['sigma']) # TODO change to model.predict
            except:
                predictions = model.predict(x_0) #linear model

        y_results = x_t_results - x_0
        if remap_angle: y_results[:,2] = np.array([_remap_angle(theta) for theta in y_results[:,2]])
        
        axs[int(round((index+1)/4,0)),index%2].plot(range_x, [y[0] for y in y_results], 'C0-', label='c_l')
        axs[int(round((index+1)/4,0)),index%2].plot(range_x, [y[1] for y in y_results], 'C1-', label='c_v')
        axs[int(round((index+1)/4,0)),index%2].plot(range_x, [y[2] for y in y_results], 'C2-', label='p_a')
        axs[int(round((index+1)/4,0)),index%2].plot(range_x, [y[3] for y in y_results], 'C3-', label='p_v')
        if model:
            axs[int(round((index+1)/4,0)),index%2].plot(range_x, [pred_[0] for pred_ in predictions], 'C0--', label='c_l_pred')
            axs[int(round((index+1)/4,0)),index%2].plot(range_x, [pred_[1] for pred_ in predictions], 'C1--', label='c_v_pred')
            axs[int(round((index+1)/4,0)),index%2].plot(range_x, [pred_[2] for pred_ in predictions], 'C2--', label='p_a_pred')
            axs[int(round((index+1)/4,0)),index%2].plot(range_x, [pred_[3] for pred_ in predictions], 'C3--', label='p_v_pred')
        axs[int(round((index+1)/4,0)),index%2].set_ylabel('component of y values')
        axs[int(round((index+1)/4,0)),index%2].set_xlabel('{} initial values'.format(index_to_variable[index]))
        axs[int(round((index+1)/4,0)),index%2].legend()

    fig.suptitle('action: {}'.format(initial_x[-1]))
    fig.tight_layout()