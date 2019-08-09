import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot():
    data = []
    #data.append([gi, [angles]])

    #data.append([42, [0.06828011734249892, 0.18915993297164044, 0.28661262567915413, 0.3358702026329062, 0.41599799247833064, 0.46314473122624017, 2.976731616692713, 3.001221905128876, 3.030543810628197, 3.0463924337562096, 3.076126454588151, 3.109956262255429]])
    #data.append([126, [0.04855161565273778, 0.1522624415120694, 0.24340410195782147, 0.3008639755673, 0.32932397338858754, 0.3522075471715323, 2.9720199564357306, 2.9872386443397705, 3.0019352050777166, 3.0200794793130656, 3.0485304751104016, 3.098398761321846]])
    data.append([163, [0.07502964424103403, 0.1766783809037154, 0.23828857108600465, 0.27338040987171475, 0.33450128417523733, 0.41318296559127193, 3.0051868904059518, 3.030141206071636, 3.0499749174672615, 3.0731456107680826, 3.0794621014292867, 3.1023482289789084]])
    data.append([191, [0.06145750160497606, 0.14355307799639958, 0.21166490910438504, 0.2641653408069171, 0.29468553512509355, 0.30753171745685826, 3.015016680933541, 3.0359910709054283, 3.0589257491688024, 3.0727908603793597, 3.091220666363035, 3.1203160696665964]])
    data.append([233, [0.0782206190242245, 0.17595984635051332, 0.23029842411277557, 0.2751521017432925, 0.30217697959865986, 0.33658545644757654, 3.012379277223728, 3.023828561985607, 3.0422288648001468, 3.0582680680395606, 3.0882100777702397, 3.122596841987013]])
    data.append([310, [0.08802626223902271, 0.20470819839894677, 0.25806028357516486, 0.31025960103982275, 0.32485226415620005, 0.3391965318358809, 3.00619904026578, 3.0177517244170358, 3.028628176743887, 3.0305533905048376, 3.0510216612111263, 3.0915483997825417]])
    data.append([411, [0.06616208376417691, 0.16574987820926038, 0.22521745716497593, 0.28342686552634533, 0.32280030276435406, 0.3644490298306171, 3.005468991323039, 3.0310814521962905, 3.0534129778646855, 3.071793951768892, 3.0914640805823606, 3.1160604328094283]])
    data.append([418, [0.08082509385270766, 0.16936732806320265, 0.2115179643563079, 0.2351117841667574, 0.27211258238394875, 0.2955044642045258, 3.0063827395478038, 3.024231365129571, 3.042758765872149, 3.0472369270815225, 3.0561459962325097, 3.099095280758593]])
    data.append([445, [0.06864014053439742, 0.1630581808501538, 0.22501324269268183, 0.29039396054670863, 0.3350527630772957, 0.3575938238819495, 3.0052225637323016, 3.0304981115757146, 3.056305437071646, 3.071608116379298, 3.0911955175622627, 3.1165011803877247]])
    data.append([476, [0.055975684977875674, 0.14617804809414242, 0.24646745148000687, 0.31025329179595806, 0.3694184144950378, 0.41964091765178285, 3.0071426255099, 3.028191769501864, 3.05862186036694, 3.087658248984562, 3.110982958347625, 3.12814879805369]])
    #data.append([480, [0.14778194212834164, 0.28863446515625646, 0.3293713337324902, 0.35123208562700853, 0.34893333652531294, 0.335155777469322, 3.029322092100559, 3.0446299610907777, 3.0502437145621095, 3.048663731760596, 3.0615311026627032, 3.091442249870395]])
    data.append([512, [0.06255495514534652, 0.14273197488362097, 0.20555376097749614, 0.27324708955659227, 0.31507855375883603, 0.3350972860696671, 3.0150119974565306, 3.03871388674034, 3.060008090804333, 3.070257774601238, 3.0879249973889573, 3.1172933215226384]])
    data.append([539, [0.08828692155764753, 0.2046492694252428, 0.29521868556230696, 0.33623037679536205, 0.3631591086214256, 0.35278215547370656, 3.0031785596478375, 3.022611943489884, 3.0384839331923787, 3.054379578352177, 3.074107574488813, 3.0992325320887586]])
    data.append([545, [0.07014825552782264, 0.1497153418358417, 0.20432080574335323, 0.23557602254365564, 0.2921868058959618, 0.32348796155961546, 3.0073381323822934, 3.0216140845519215, 3.04487029160981, 3.0452025348852354, 3.0712065449364476, 3.0983908205004993]])
    data.append([558, [0.0597266538797419, 0.14668959203994736, 0.21313158138515578, 0.25825450412222306, 0.2833435228859431, 0.3257896015807834, 3.0138767588014312, 3.0302325827663914, 3.052130774760815, 3.0719098360447155, 3.090136088019847, 3.1190410711871146]])
    data.append([562, [0.08168117782810282, 0.1677853738211348, 0.22165683667479436, 0.2859928453822903, 0.3132731477766016, 0.3528655563400221, 3.005172016174299, 3.024228938950495, 3.043805510263642, 3.058038448582309, 3.0771206504565303, 3.107608667765528]])
    #data.append([577, [0.1127559722371163, 0.19799910484054814, 0.2190724368074649, 0.2807184304674828, 0.3289837487759858, 0.31843697558857004, 3.021886047783121, 3.0467879247332297, 3.0695782001296164, 3.061953608630488, 3.0616550173482255, 3.083969033444342]])
    #data.append([584, [0.06046071747604813, 0.14224540513567038, 0.18129750124953045, 0.2378848500836773, 0.29296853582682086, 0.31276548556390216, 3.010328845740839, 3.0366226817751985, 3.0590300283960965, 3.05650623726022, 3.0743891214075996, 3.112593907504802]])
    #data.append([608, [0.07301310597580504, 0.07508707024705173, 0.027495434853676413, 0.120880957921718, 0.20912277661809303, 0.32888057848480134, 2.9900853848625255, 2.9, 2.9967472173579255, 3.0531932755643325, 3.0871754661761703, 3.101534381334683]])
    data.append([615, [0.08376609077766523, 0.2063483612830859, 0.2900447747763091, 0.3372263914196495, 0.33702020900540985, 0.3192998280326423, 3.0092939454202456, 3.015545282258144, 3.0317385223284337, 3.045795149620273, 3.066955228681273, 3.108972446083074]])
    #data.append([628, [0.18510790890868126, 0.33509412309521003, 0.37229912836096163, 0.3762820672012744, 0.3760117191491491, 0.37863474871435815, 3.0045983662101143, 3.001297020713763, 3.001340772733977, 3.01741158605107, 3.044210339970217, 3.087673712865664]])
    data.append([664, [0.0394859147334874, 0.12263776121647131, 0.1967312314069248, 0.2576968557466459, 0.30254848756801334, 0.3384539219594298, 2.9951760196604353, 3.018608487261139, 3.043261885894211, 3.06108085983458, 3.0888626094878373, 3.11796965995428]])
    #data.append([703, [0.102334868570011, 0.17907916444589775, 0.2299034353175787, 0.27240594005087476, 0.3137242326610002, 0.32381131874924296, 2.999389617906961, 3.029446734112088, 3.063310275698269, 3.0463967153843874, 3.0498186740468998, 3.0776977744753555]])
    data.append([707, [0.06637060658270008, 0.16196931040708784, 0.2200854354186483, 0.26668710714579746, 0.2971915690734514, 0.3147424570423236, 3.0114038373994103, 3.0214189188762597, 3.0381766363414022, 3.053023962919584, 3.082680330902792, 3.1196583748154936]])
    data.append([709, [0.10347344984103173, 0.20630937729352916, 0.275586508056729, 0.30202159655348754, 0.31194937461398337, 0.3064393934912712, 3.008253285344874, 3.0208230628306785, 3.0400692945431937, 3.0427107761159076, 3.0534819821333605, 3.0907815922596007]])
    data.append([710, [0.040015271878073515, 0.12531025552088512, 0.21278834737032878, 0.2855066799673787, 0.33559391752098655, 0.3846146244290118, 3.016107884138974, 3.0299661723437095, 3.0513048263748987, 3.0829431072843336, 3.111106596634183, 3.1310245518315414]])
    data.append([764, [0.0676359824843819, 0.1530589993007193, 0.2196833265202975, 0.24742813153582646, 0.2822441206127299, 0.29132424019805486, 3.007234318347253, 3.0311493027737377, 3.051529094107867, 3.0542297262804645, 3.0607565403421773, 3.0960186045831493]]) 
    #data.append([787, [0.055616912846171804, 0.1331900292349522, 0.19454471329615639, 0.23051831460079136, 0.2527642279595668, 0.40213627520462986, 2.9947567260326258, 3.009148231975895, 3.0164286849717312, 3.0241135538178265, 3.0788667764917603, 3.141592653589793]])
    data.append([804, [0.05564277347746274, 0.1514313318572252, 0.21096362609574165, 0.2560078514913094, 0.30347277710441795, 0.3320876739629853, 2.9978031064249695, 3.0082997730105414, 3.0290978555688595, 3.045398375141104, 3.081287721852296, 3.1145597279149837]])
    data.append([812, [0.046309735994043116, 0.1406400328095235, 0.22936849859761865, 0.2901464793382182, 0.31632761502983026, 0.3435033008891132, 2.998554915783327, 3.018040595476435, 3.039413695390037, 3.0631972293514402, 3.0956052135778878, 3.126084513686262]])
    data.append([823, [0.06701499413069492, 0.1479646444988235, 0.21711241551242944, 0.2879906610789483, 0.31818374270610017, 0.3757796482203077, 3.007658489107549, 3.031382409839353, 3.050664430554643, 3.0748342241322635, 3.0968320285803026, 3.1202158383456884]])
    data.append([831, [0.07064955638596618, 0.17354811865191916, 0.23349164159346247, 0.269343467172704, 0.33842617251666257, 0.4334554631050112, 3.0205984053694626, 3.042387225007543, 3.0617725728406953, 3.079186933733301, 3.094469717657064, 3.1231158411117406]])
    data.append([863, [0.05741791333026321, 0.15518717755001088, 0.23567689275094605, 0.2658970494441248, 0.2914145037801215, 0.28544353600509126, 2.999682667664419, 3.0116977781599967, 3.025441028865861, 3.0454364745340436, 3.068115595669086, 3.108148063206169]])
    data.append([873, [0.08952391363741213, 0.20041139618926626, 0.26463245599623186, 0.29893097390879353, 0.31549655196965215, 0.38509938190162113, 3.010506777878533, 3.03857937475657, 3.0545364898337746, 3.07407019321076, 3.078426691969903, 3.098190259709498]])
    data.append([888, [0.05874023426288838, 0.14859238572011674, 0.21604574470623675, 0.2642037683976231, 0.3091920052475787, 0.3677854619166727, 3.005564426302645, 3.025160319553319, 3.0497957150196466, 3.0706685818982176, 3.0845528176649473, 3.1158683219633154]])
    data.append([897, [0.07505258185902974, 0.1857716193398962, 0.25997538393675246, 0.3177764905421256, 0.3276705498583743, 0.35205291209506917, 3.015080820977564, 3.027106840278695, 3.035492012057103, 3.052028160188438, 3.0773824666819465, 3.1147661461426455]])
    #data.append([902, [0.08952415181684524, 0.20980726350424836, 0.26749570331206574, 0.29519444997072825, 0.3381712262849823, 0.37837592225296507, 3.0144707130884436, 3.0296728147206977, 3.048365009216258, 3.0629295620272963, 3.059994977184354, 3.0940990334853318]])
               
    #for i in range(len(data)): plt.plot([1,2,3,4,5,6], data[i][1][:6], '-o', label=str(data[i][0]))

    
    for i in range(len(data)): 
        plt.plot([1,2,3,4,5,6], data[i][1][6:], '-o', label=str(data[i][0]))
        print('gi = ' + str(data[i][0]))
        
    #plt.legend()
    #plt.gca().set_ylim([0.89, 1])
    #plt.xticks(np.arange(1, , step=1))
    plt.xlabel('$\\beta_i$')
    plt.ylabel('Value of $\\beta_i$')
    plt.show()
    
if __name__ == '__main__':
    plot()
