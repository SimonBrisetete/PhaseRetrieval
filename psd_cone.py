import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import plotly.graph_objects as go

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = np.arange(0, 2, 0.02)
y = np.arange(0, 2, 0.02)

#set(gcf,'color','w')
X, Y= np.meshgrid(x,y)
#Z = np.sqrt(np.dot(X, Y))
Z = a*X**2 + b*Y**2

fig = go.Figure(
    data=[go.Surface(z=Z, x=x, y=y, colorscale="Reds", opacity=0.5)])
fig.update_layout(
    title='My title',
    autosize=False,
    width=500,
    height=500,
    margin=dict(l=65, r=50, b=65, t=90),
    scene_aspectmode='cube'
)
fig.show()

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,
#                        linewidth=0, antialiased=False)
# # Customize the axis.
# ax.set_xlim(0, 2)
# ax.set_ylim(0, 2)
# ax.set_zlim(-2, 2)
#
#
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# #ax.zaxis.set_major_formatter('{x:.02f}')
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.view(62, 64)
# plt.show()
# #
# # #axis([0,2,0,2,-2,2]);
# # surf(X,Y, Z,'edgecolor','none');shading interp
# # surf(X,Y,-Z,'edgecolor','none');shading interp
# #
# for t=0:0.2:1:
#     ax.p([0,2],[0,2*t^2],[0,2*t],'linewidth',1,'color','k');
#     plot3([0,2*t^2],[0,2],[0,2*t],'linewidth',1,'color','k');

