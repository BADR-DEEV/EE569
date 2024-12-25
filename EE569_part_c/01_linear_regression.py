import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def F(w_0, w_1):
    return w_0 + w_1 * t

def E(w_0, w_1):
    return np.linalg.norm(np.array([t,np.ones_like(t)]).transpose().dot(np.array([w_0,w_1]))-y)

axis_color = 'lightgoldenrodyellow'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.subplots_adjust(left=0.05, bottom=0.25)

w_0 = 2
w_1 = 3
t = np.arange(-1.0, 1.1, 0.2)
y_true = w_0 + w_1 * t
noise = np.random.normal(0, 2, t.shape)
y = y_true + noise



# Plot on ax1
ax1.scatter(t, y, label='Noisy observations', color='green')
ax1.plot(t, y_true, label='Ground truth', color='green')
[line1] = ax1.plot(t, F(w_0, w_1), linewidth=2, color='red')
ax1.set_xlim([-1, 1])
ax1.set_ylim([-10, 10])

# Plot on ax2
w_s = np.arange(-10.0, 10.0, 0.1)
E_s = [E(w_0,w_i) for w_i in w_s]
ax2.plot(w_s, E_s, label='Error curve', color='black')
[line2] = ax2.plot(w_1, E(w_0, w_1),color='orange', marker='o', markersize=12)

# Add sliders
w_0_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
w_0_slider = Slider(w_0_slider_ax, 'w_0', -10.0, 10.0, valinit=w_0)

w_1_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
w_1_slider = Slider(w_1_slider_ax, 'w_1', -10.0, 10.0, valinit=w_1)

# Update both plots when sliders change
def sliders_on_changed(val):
    line1.set_ydata(F(w_0_slider.val, w_1_slider.val))
    line2.set_ydata(E(w_0_slider.val, w_1_slider.val))
    line2.set_xdata( w_1_slider.val)
    fig.canvas.draw_idle()

w_0_slider.on_changed(sliders_on_changed)
w_1_slider.on_changed(sliders_on_changed)

# Add reset button
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')

def reset_button_on_clicked(mouse_event):
    w_0_slider.reset()
    w_1_slider.reset()

reset_button.on_clicked(reset_button_on_clicked)
plt.legend(fontsize="large")
plt.show()