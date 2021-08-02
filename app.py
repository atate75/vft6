from flask.scaffold import *
from flask import request, jsonify, render_template, Flask
from matplotlib import pyplot as plt
import os, random
import datetime, statistics
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from flask_cors import CORS, cross_origin
from matplotlib import patches
from matplotlib import colors
app = Flask(__name__)
CORS(app)

def return_dots_VFT6(dataset, reliability_dict): #Results/analysis 
    x_good = []; y_good = []; x_bad =[]; y_bad =[]; color=[]; color_good = [];now = datetime.datetime.now()
    time_set = []; time_set_GOOD = []; time_set_GOODx = []; time_set_badX = []
    crt = 0
    x_vals = [];z = [];z2 = [];big_data = []

    for point in dataset: #good-bad, x,y,color, time
        if point[0] == 0:
            time_set_badX.append(crt)
            x_bad.append(point[1])
            y_bad.append(point[2])
            color.append(point[3])
            if point[3] >= 79:
                x_vals.append([point[1],point[2]])
                z.append(0)
        else:
            time_set_GOOD.append(point[-1])
            x_good.append(point[1])
            y_good.append(point[2])
            color_good.append(point[3])
            x_vals.append([point[1],point[2]])
            z.append(1)
            z2.append(point[3])
            big_data.append([point[1],point[2]])

        time_set_GOODx.append(crt)
        time_set.append(point[-1])
        crt += 1
  
    
    upper_bound_time = statistics.median(time_set_GOOD) + 2*statistics.stdev(time_set_GOOD)

    FPR = str(reliability_dict["FPR"][1]-reliability_dict["FPR"][0]) +" / "+str(reliability_dict["FPR"][1])
    FNR = str(reliability_dict["FPR"][1]-reliability_dict["FNR"][0]) +" / "+str(reliability_dict["FNR"][1])
    number_dots = len(x_good) + len(x_bad)
    time_now = now.strftime("%Y-%m-%d")

    sdr = 0
    locations = []
    crt = 0

    for element in dataset:
        if element[-1] < 1400 and (element[-1] > upper_bound_time or element[-1] <100):
            sdr += 1
            locations.append(crt)
        crt+= 1
    sdr /= len(dataset)


    fig = plt.figure()
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

    spec = gridspec.GridSpec(ncols = 4, nrows =5,figure=fig)


    ax1 = fig.add_subplot(spec[0:2, 0:2])
    ax2 = fig.add_subplot(spec[-1,:])
    ax3 = fig.add_subplot(spec[0:2, 2:4])
    ax4 = fig.add_subplot(spec[2:4, 0:2])
    ax5 = fig.add_subplot(spec[2:4, 2:4])

    fig.suptitle(f"AT-VFT6 BETA ({time_now})",fontsize=18,fontweight="bold")
    fig.set_size_inches(w=8.5,h=11)
    

    ax1.scatter(x_good, y_good, s=60,color="g")
    ax1.scatter(x_bad, y_bad, s=60, c=color, edgecolors="red",cmap="gray")
    ax1.set_facecolor("#252525")
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='white')
    ax1.axvline(x=0, color='white')
    ax1.set_title("Raw Data")

    
    ax2.plot(time_set_GOODx,time_set, label="Response time")
    ax2.axhline(y=upper_bound_time,color="g", linestyle="--", label="Upper Bound Outliers")
    ax2.axhline(y=upper_bound_time,color="m", linestyle="--", label="Moving Median")
    ax2.legend()
    ax2.set_ylabel("ms")
    ax2.set_title("Reliability Metrics")
    ax2.grid(True)
    for i in time_set_badX:
        ax2.axvspan(i,i+1,color="red",alpha=0.25, label="Failed Dots")
    for i in locations:
        ax2.axvspan(i,i+1,color="blue",alpha=0.25, linestyle="--" )
    

    nbhrs = KNeighborsClassifier(n_neighbors=7, weights='distance')
    nbhrs.fit(x_vals,z)
    
    y_range = 500
    x_range = 500
    xx, yy = np.meshgrid(np.linspace(-1*x_range, x_range, 50), np.linspace(-1*y_range, y_range, 50))
  
    prediction = nbhrs.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,0]
    prediction = prediction.reshape(xx.shape)

    ax4.set_title("Predicted Loss")
    nbhrs_contrast_1 = KNeighborsClassifier(n_neighbors=3, weights='distance')
    nbhrs_contrast_1.fit(big_data, z2)
    
    prediction_contrast_1 = nbhrs_contrast_1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax4.set_aspect('equal')
    
    
    
    ax1.set_xlim([-500,500])
    ax1.set_ylim([-500,500])
    ax3.pcolormesh(prediction,cmap='binary')
    ax3.add_patch(patches.Circle((25,25),radius=37,color='white',linewidth=100, fill = False))
    ax4.add_patch(patches.Circle((25,25),radius=37,color='white',linewidth=100, fill = False))
    ax1.add_patch(patches.Circle((0,0),radius=710,color='white',linewidth=100, fill = False))


    locations = ['top','right','bottom','left']
    axes = [ax1,ax3,ax4,ax5]
    for loc in locations:
        for a in axes:
            a.spines[loc].set_visible(False)

    for a in axes:
        a.xaxis.set_ticklabels([])
        a.xaxis.set_ticks_position('none')
        a.yaxis.set_ticks_position('none')
        a.yaxis.set_ticklabels([])

    ax3.set_xlim([0,50])
    ax3.set_ylim([0,50])
    axes = [ax3,ax4,ax5]
    for a in axes:
        a.set_xlim([0,50])
        a.set_ylim([0,50])
        a.axhline(y=25,alpha = 0.5, color='k')
        a.axvline(x=25,alpha = 0.5, color='k')

    ax3_score = []
    ax5_score = []
    for (i,j),z in np.ndenumerate(prediction):
        var = int((10-z)*3)
        ax3_score.append(var)
        if i%9 == 0 and j%9 == 0:
            if ((i-23)**2+(j-24)**2)**0.5 < 24.9:
                ax3.text(j+2,i+1,str(var),ha='left',va='bottom', bbox=dict(boxstyle='round',facecolor='white',edgecolor='0.3'))
            

    for (i,j),z in np.ndenumerate(prediction_contrast_1):
        var = int((100-z)*9/24)
        ax5_score.append(var)
        if i%5 == 0 and j%5 == 0:
            if ((i-23)**2+(j-24)**2)**0.5 < 24.9:
                
                ax5.text(j+1,i+1,str(var),ha='left',va='bottom', bbox=dict(boxstyle='round',facecolor='white',edgecolor='0.3'))
                

    ax3_score = round(statistics.mean(ax3_score),1)
    ax5_score = round(statistics.mean(ax5_score),1)
    ax3.set_title(f"BlindSpotProb [{ax3_score}]")
    ax5.set_title(f"Predicted Loss [{ax5_score}]")
    cmap = colors.ListedColormap(['white', 'red'])
    bounds=[0,0.5,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)



    prediction[prediction > 0.75] = 1
    prediction[prediction < 0.75] = 0
    ax4.pcolormesh(prediction,cmap=cmap, norm=norm)
    ax4.pcolormesh(prediction_contrast_1,alpha=0.7,cmap=plt.cm.get_cmap("binary",7),vmin=20,vmax=100)
    
    tt = convert(reliability_dict["TotalTime"]/1000)
    
    statement = f"Dots: {number_dots}, FalsePOS ERR: {FPR}, FalseNEG ERR: {FNR}, Gaze EST: {round(sdr,2)} Total Time: {tt}."
    ax2.set_xlabel("Dot \n " + statement,fontweight="bold")
    ax3.set_aspect('equal')


    fig.tight_layout()

    return fig

def convert(seconds):
    seconds = seconds % (24 * 3600)
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%02d:%02d" % (minutes, seconds)

@app.route("/")
def main():
    return render_template("VFT6.html")
 
@app.route("/process", methods=['GET','POST','OPTIONS'])
@cross_origin()
def processjson():
    
    data = request.get_json(force=True)
    DataSet = data['DataSet']
    reliability_dict = data['reliability_dict']
    fig = return_dots_VFT6(DataSet,reliability_dict)
    cwd = os.getcwd()
    
    stuff_to_remove = os.listdir(cwd+"/static")

    for i in stuff_to_remove:
        os.remove(cwd+"/static/"+i)

    path = cwd + "/static/"
    name1 = str(random.random()) + ".pdf"
    fig.savefig(path+name1)
    response = jsonify({"redirect": "/static/"+name1})
    return response
    

