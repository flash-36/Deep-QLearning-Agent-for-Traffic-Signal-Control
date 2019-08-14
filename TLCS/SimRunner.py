import traci
import numpy as np
import random

# phase codes based on tlcs.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

# HANDLE THE SIMULATION OF THE AGENT
class SimRunner:
    def __init__(self, sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd, demo=False):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._traffic_gen = traffic_gen
        self._total_episodes = total_episodes
        self._gamma = gamma
        self._eps = 0  # controls the explorative/exploitative payoff, I chose epsilon-greedy policy
        self._steps = 0
        self._waiting_times = {}
        self._sumoCmd = sumoCmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._sum_intersection_queue = 0
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_intersection_queue_store = []
        self._demo=demo


    # THE MAIN FUCNTION WHERE THE SIMULATION HAPPENS
    def run(self, episode):
        # first, generate the route file for this simulation and set up sumo
        if not self._demo:
            self._traffic_gen.generate_routefile(episode)
        
        if self._demo:
            with open("intersection/tlcs_train.rou.xml", "w") as routes:
            print("""<routes>
              <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

              <route id="W_N" edges="W2TL TL2N"/>
              <route id="W_E" edges="W2TL TL2E"/>
              <route id="W_S" edges="W2TL TL2S"/>
              <route id="N_W" edges="N2TL TL2W"/>
              <route id="N_E" edges="N2TL TL2E"/>
              <route id="N_S" edges="N2TL TL2S"/>
              <route id="E_W" edges="E2TL TL2W"/>
              <route id="E_N" edges="E2TL TL2N"/>
              <route id="E_S" edges="E2TL TL2S"/>
              <route id="S_W" edges="S2TL TL2W"/>
              <route id="S_N" edges="S2TL TL2N"/>
              <route id="S_E" edges="S2TL TL2E"/>
            <vehicle id="S_E_0" type="standard_car" route="S_E" depart="18.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_1" type="standard_car" route="E_W" depart="53.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_2" type="standard_car" route="S_N" depart="70.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_3" type="standard_car" route="N_S" depart="70.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_4" type="standard_car" route="E_W" depart="71.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_5" type="standard_car" route="S_N" depart="76.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_6" type="standard_car" route="W_E" depart="79.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_7" type="standard_car" route="E_W" depart="96.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_8" type="standard_car" route="S_E" depart="100.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_9" type="standard_car" route="W_E" depart="106.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_10" type="standard_car" route="E_S" depart="111.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_11" type="standard_car" route="S_E" depart="111.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_12" type="standard_car" route="S_W" depart="112.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_13" type="standard_car" route="N_W" depart="113.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_14" type="standard_car" route="N_S" depart="113.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_15" type="standard_car" route="E_S" depart="118.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_16" type="standard_car" route="E_W" depart="122.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_17" type="standard_car" route="W_E" depart="124.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_18" type="standard_car" route="N_S" depart="131.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_19" type="standard_car" route="E_N" depart="142.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_20" type="standard_car" route="N_S" depart="143.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_21" type="standard_car" route="S_N" depart="149.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_22" type="standard_car" route="S_E" depart="149.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_23" type="standard_car" route="S_E" depart="153.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_24" type="standard_car" route="S_N" depart="155.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_25" type="standard_car" route="E_S" depart="160.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_26" type="standard_car" route="W_E" depart="162.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_27" type="standard_car" route="W_E" depart="163.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_28" type="standard_car" route="W_N" depart="170.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_29" type="standard_car" route="N_E" depart="172.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_30" type="standard_car" route="S_N" depart="177.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_31" type="standard_car" route="E_W" depart="179.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_32" type="standard_car" route="S_N" depart="186.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_33" type="standard_car" route="N_E" depart="188.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_34" type="standard_car" route="W_E" depart="189.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_35" type="standard_car" route="N_E" depart="190.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_36" type="standard_car" route="E_W" depart="193.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_37" type="standard_car" route="N_E" depart="193.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_38" type="standard_car" route="S_W" depart="194.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_39" type="standard_car" route="E_W" depart="194.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_40" type="standard_car" route="N_S" depart="196.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_41" type="standard_car" route="E_W" depart="202.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_42" type="standard_car" route="W_E" depart="203.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_43" type="standard_car" route="E_W" depart="207.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_44" type="standard_car" route="N_E" depart="208.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_45" type="standard_car" route="S_N" depart="210.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_46" type="standard_car" route="W_E" depart="212.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_47" type="standard_car" route="E_W" depart="212.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_48" type="standard_car" route="E_W" depart="215.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_49" type="standard_car" route="W_E" depart="218.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_50" type="standard_car" route="S_N" depart="219.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_51" type="standard_car" route="N_E" depart="223.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_52" type="standard_car" route="E_W" depart="223.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_53" type="standard_car" route="N_W" depart="231.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_54" type="standard_car" route="W_S" depart="231.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_55" type="standard_car" route="N_S" depart="237.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_56" type="standard_car" route="S_W" depart="237.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_57" type="standard_car" route="S_N" depart="237.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_58" type="standard_car" route="E_W" depart="238.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_59" type="standard_car" route="W_S" depart="240.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_60" type="standard_car" route="N_W" depart="244.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_61" type="standard_car" route="E_S" depart="249.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_62" type="standard_car" route="W_E" depart="250.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_63" type="standard_car" route="E_W" depart="252.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_64" type="standard_car" route="W_S" depart="254.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_65" type="standard_car" route="W_E" depart="257.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_66" type="standard_car" route="W_N" depart="258.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_67" type="standard_car" route="W_E" depart="260.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_68" type="standard_car" route="S_N" depart="261.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_69" type="standard_car" route="N_S" depart="263.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_70" type="standard_car" route="W_E" depart="263.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_71" type="standard_car" route="S_W" depart="267.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_72" type="standard_car" route="N_E" depart="272.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_73" type="standard_car" route="S_N" depart="272.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_74" type="standard_car" route="W_S" depart="274.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_75" type="standard_car" route="S_N" depart="280.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_76" type="standard_car" route="E_W" depart="283.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_77" type="standard_car" route="S_W" depart="285.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_78" type="standard_car" route="E_W" depart="286.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_79" type="standard_car" route="N_S" depart="287.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_80" type="standard_car" route="W_E" depart="288.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_81" type="standard_car" route="W_E" depart="289.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_82" type="standard_car" route="S_N" depart="290.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_83" type="standard_car" route="W_E" depart="296.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_84" type="standard_car" route="E_S" depart="299.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_85" type="standard_car" route="E_W" depart="299.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_86" type="standard_car" route="N_S" depart="300.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_87" type="standard_car" route="N_S" depart="301.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_88" type="standard_car" route="E_W" depart="301.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_89" type="standard_car" route="W_E" depart="302.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_90" type="standard_car" route="N_E" depart="305.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_91" type="standard_car" route="W_E" depart="308.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_92" type="standard_car" route="W_E" depart="308.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_93" type="standard_car" route="W_E" depart="309.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_94" type="standard_car" route="E_S" depart="311.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_95" type="standard_car" route="S_N" depart="312.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_96" type="standard_car" route="S_W" depart="312.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_97" type="standard_car" route="N_S" depart="312.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_98" type="standard_car" route="W_N" depart="313.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_99" type="standard_car" route="N_W" depart="313.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_100" type="standard_car" route="W_E" depart="315.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_101" type="standard_car" route="S_N" depart="316.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_102" type="standard_car" route="E_W" depart="317.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_103" type="standard_car" route="N_S" depart="318.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_104" type="standard_car" route="E_W" depart="318.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_105" type="standard_car" route="W_E" depart="319.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_106" type="standard_car" route="W_E" depart="320.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_107" type="standard_car" route="E_N" depart="323.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_108" type="standard_car" route="S_W" depart="323.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_109" type="standard_car" route="W_N" depart="329.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_110" type="standard_car" route="N_S" depart="331.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_111" type="standard_car" route="N_S" depart="331.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_112" type="standard_car" route="W_E" depart="333.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_113" type="standard_car" route="N_W" depart="333.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_114" type="standard_car" route="N_W" depart="334.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_115" type="standard_car" route="S_N" depart="335.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_116" type="standard_car" route="N_S" depart="339.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_117" type="standard_car" route="S_E" depart="340.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_118" type="standard_car" route="N_S" depart="340.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_119" type="standard_car" route="W_E" depart="343.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_120" type="standard_car" route="N_W" depart="345.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_121" type="standard_car" route="E_W" depart="346.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_122" type="standard_car" route="W_E" depart="347.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_123" type="standard_car" route="E_W" depart="348.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_124" type="standard_car" route="W_N" depart="356.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_125" type="standard_car" route="E_S" depart="358.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_126" type="standard_car" route="W_N" depart="358.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_127" type="standard_car" route="E_S" depart="360.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_128" type="standard_car" route="E_W" depart="360.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_129" type="standard_car" route="N_E" depart="361.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_130" type="standard_car" route="W_E" depart="367.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_131" type="standard_car" route="S_E" depart="368.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_132" type="standard_car" route="N_S" depart="370.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_133" type="standard_car" route="N_S" depart="371.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_134" type="standard_car" route="W_E" depart="371.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_135" type="standard_car" route="N_S" depart="374.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_136" type="standard_car" route="W_E" depart="376.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_137" type="standard_car" route="S_N" depart="376.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_138" type="standard_car" route="N_S" depart="377.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_139" type="standard_car" route="E_N" depart="378.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_140" type="standard_car" route="S_E" depart="378.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_141" type="standard_car" route="W_E" depart="379.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_142" type="standard_car" route="N_E" depart="380.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_143" type="standard_car" route="E_S" depart="380.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_144" type="standard_car" route="S_N" depart="381.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_145" type="standard_car" route="E_W" depart="384.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_146" type="standard_car" route="S_N" depart="384.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_147" type="standard_car" route="E_W" depart="386.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_148" type="standard_car" route="E_W" depart="388.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_149" type="standard_car" route="W_E" depart="391.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_150" type="standard_car" route="N_W" depart="392.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_151" type="standard_car" route="W_N" depart="399.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_152" type="standard_car" route="S_N" depart="401.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_153" type="standard_car" route="S_N" depart="402.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_154" type="standard_car" route="E_W" depart="405.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_155" type="standard_car" route="S_N" depart="405.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_156" type="standard_car" route="N_W" depart="405.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_157" type="standard_car" route="W_E" depart="407.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_158" type="standard_car" route="W_E" depart="408.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_159" type="standard_car" route="W_E" depart="412.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_160" type="standard_car" route="W_E" depart="412.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_161" type="standard_car" route="W_E" depart="413.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_162" type="standard_car" route="S_E" depart="413.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_163" type="standard_car" route="E_W" depart="420.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_164" type="standard_car" route="W_E" depart="421.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_165" type="standard_car" route="S_W" depart="422.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_166" type="standard_car" route="N_S" depart="424.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_167" type="standard_car" route="S_W" depart="424.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_168" type="standard_car" route="S_N" depart="425.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_169" type="standard_car" route="S_W" depart="427.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_170" type="standard_car" route="W_E" depart="427.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_171" type="standard_car" route="N_S" depart="428.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_172" type="standard_car" route="N_E" depart="432.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_173" type="standard_car" route="N_S" depart="433.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_174" type="standard_car" route="N_E" depart="436.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_175" type="standard_car" route="W_E" depart="439.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_176" type="standard_car" route="S_W" depart="443.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_177" type="standard_car" route="E_W" depart="445.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_178" type="standard_car" route="E_N" depart="448.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_179" type="standard_car" route="E_W" depart="449.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_180" type="standard_car" route="S_N" depart="450.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_181" type="standard_car" route="W_S" depart="451.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_182" type="standard_car" route="W_S" depart="456.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_183" type="standard_car" route="W_E" depart="457.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_184" type="standard_car" route="S_N" depart="457.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_185" type="standard_car" route="N_S" depart="460.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_186" type="standard_car" route="E_W" depart="463.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_187" type="standard_car" route="N_S" depart="464.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_188" type="standard_car" route="S_E" depart="465.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_189" type="standard_car" route="N_S" depart="465.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_190" type="standard_car" route="W_E" depart="465.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_191" type="standard_car" route="E_N" depart="466.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_192" type="standard_car" route="W_E" depart="468.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_193" type="standard_car" route="N_E" depart="468.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_194" type="standard_car" route="S_E" depart="468.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_195" type="standard_car" route="E_S" depart="469.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_196" type="standard_car" route="W_N" depart="470.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_197" type="standard_car" route="W_S" depart="472.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_198" type="standard_car" route="E_W" depart="472.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_199" type="standard_car" route="W_E" depart="473.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_200" type="standard_car" route="N_S" depart="476.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_201" type="standard_car" route="S_N" depart="477.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_202" type="standard_car" route="W_E" depart="480.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_203" type="standard_car" route="E_W" depart="481.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_204" type="standard_car" route="N_S" depart="485.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_205" type="standard_car" route="S_E" depart="487.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_206" type="standard_car" route="N_S" depart="489.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_207" type="standard_car" route="W_S" depart="493.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_208" type="standard_car" route="S_W" depart="493.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_209" type="standard_car" route="S_E" depart="498.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_210" type="standard_car" route="N_W" depart="498.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_211" type="standard_car" route="W_N" depart="501.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_212" type="standard_car" route="W_E" depart="502.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_213" type="standard_car" route="W_E" depart="504.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_214" type="standard_car" route="S_N" depart="504.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_215" type="standard_car" route="W_E" depart="504.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_216" type="standard_car" route="S_E" depart="505.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_217" type="standard_car" route="E_W" depart="505.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_218" type="standard_car" route="E_S" depart="506.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_219" type="standard_car" route="S_W" depart="507.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_220" type="standard_car" route="E_W" depart="508.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_221" type="standard_car" route="N_S" depart="510.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_222" type="standard_car" route="N_S" depart="511.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_223" type="standard_car" route="S_N" depart="512.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_224" type="standard_car" route="N_E" depart="512.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_225" type="standard_car" route="S_E" depart="512.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_226" type="standard_car" route="E_W" depart="513.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_227" type="standard_car" route="E_W" depart="516.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_228" type="standard_car" route="S_W" depart="518.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_229" type="standard_car" route="W_S" depart="518.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_230" type="standard_car" route="N_S" depart="520.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_231" type="standard_car" route="N_S" depart="523.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_232" type="standard_car" route="S_W" depart="524.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_233" type="standard_car" route="E_N" depart="525.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_234" type="standard_car" route="N_S" depart="527.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_235" type="standard_car" route="N_S" depart="528.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_236" type="standard_car" route="W_N" depart="530.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_237" type="standard_car" route="N_S" depart="532.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_238" type="standard_car" route="W_N" depart="532.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_239" type="standard_car" route="W_N" depart="534.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_240" type="standard_car" route="W_S" depart="537.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_241" type="standard_car" route="N_E" depart="541.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_242" type="standard_car" route="S_N" depart="542.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_243" type="standard_car" route="E_W" depart="543.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_244" type="standard_car" route="E_W" depart="545.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_245" type="standard_car" route="W_E" depart="547.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_246" type="standard_car" route="S_E" depart="548.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_247" type="standard_car" route="N_W" depart="551.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_248" type="standard_car" route="N_S" depart="553.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_249" type="standard_car" route="S_N" depart="553.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_250" type="standard_car" route="N_E" depart="557.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_251" type="standard_car" route="S_W" depart="559.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_252" type="standard_car" route="W_N" depart="559.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_253" type="standard_car" route="S_N" depart="561.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_254" type="standard_car" route="N_W" depart="561.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_255" type="standard_car" route="S_N" depart="563.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_256" type="standard_car" route="N_E" depart="566.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_257" type="standard_car" route="N_S" depart="567.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_258" type="standard_car" route="E_W" depart="569.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_259" type="standard_car" route="E_S" depart="572.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_260" type="standard_car" route="W_E" depart="573.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_261" type="standard_car" route="N_S" depart="574.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_262" type="standard_car" route="W_E" depart="575.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_263" type="standard_car" route="N_S" depart="576.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_264" type="standard_car" route="E_N" depart="576.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_265" type="standard_car" route="S_W" depart="580.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_266" type="standard_car" route="S_N" depart="581.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_267" type="standard_car" route="W_N" depart="584.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_268" type="standard_car" route="E_W" depart="585.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_269" type="standard_car" route="W_E" depart="585.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_270" type="standard_car" route="E_S" depart="585.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_271" type="standard_car" route="W_E" depart="588.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_272" type="standard_car" route="N_S" depart="589.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_273" type="standard_car" route="N_S" depart="592.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_274" type="standard_car" route="W_N" depart="599.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_275" type="standard_car" route="S_E" depart="600.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_276" type="standard_car" route="E_W" depart="600.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_277" type="standard_car" route="S_E" depart="604.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_278" type="standard_car" route="E_W" depart="606.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_279" type="standard_car" route="E_S" depart="609.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_280" type="standard_car" route="W_S" depart="611.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_281" type="standard_car" route="N_S" depart="616.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_282" type="standard_car" route="N_S" depart="616.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_283" type="standard_car" route="S_N" depart="619.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_284" type="standard_car" route="N_S" depart="628.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_285" type="standard_car" route="N_E" depart="629.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_286" type="standard_car" route="S_E" depart="633.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_287" type="standard_car" route="E_W" depart="633.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_288" type="standard_car" route="W_E" depart="635.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_289" type="standard_car" route="S_N" depart="638.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_290" type="standard_car" route="N_W" depart="638.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_291" type="standard_car" route="E_W" depart="640.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_292" type="standard_car" route="S_N" depart="641.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_293" type="standard_car" route="S_E" depart="642.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_294" type="standard_car" route="N_W" depart="643.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_295" type="standard_car" route="S_N" depart="647.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_296" type="standard_car" route="W_E" depart="652.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_297" type="standard_car" route="S_N" depart="655.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_298" type="standard_car" route="N_E" depart="657.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_299" type="standard_car" route="E_N" depart="658.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_300" type="standard_car" route="N_S" depart="658.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_301" type="standard_car" route="W_S" depart="661.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_302" type="standard_car" route="W_E" depart="663.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_303" type="standard_car" route="E_W" depart="665.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_304" type="standard_car" route="W_E" depart="670.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_305" type="standard_car" route="W_E" depart="671.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_306" type="standard_car" route="S_W" depart="671.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_307" type="standard_car" route="N_S" depart="674.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_308" type="standard_car" route="S_N" depart="680.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_309" type="standard_car" route="E_N" depart="681.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_310" type="standard_car" route="S_E" depart="684.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_311" type="standard_car" route="S_N" depart="685.0" departLane="random" departSpeed="10" />
            <vehicle id="W_N_312" type="standard_car" route="W_N" depart="686.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_313" type="standard_car" route="S_E" depart="690.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_314" type="standard_car" route="N_S" depart="692.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_315" type="standard_car" route="E_W" depart="700.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_316" type="standard_car" route="N_S" depart="701.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_317" type="standard_car" route="W_E" depart="706.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_318" type="standard_car" route="S_W" depart="708.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_319" type="standard_car" route="S_N" depart="711.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_320" type="standard_car" route="N_E" depart="712.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_321" type="standard_car" route="S_N" depart="713.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_322" type="standard_car" route="W_E" depart="714.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_323" type="standard_car" route="W_E" depart="717.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_324" type="standard_car" route="W_S" depart="718.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_325" type="standard_car" route="E_W" depart="722.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_326" type="standard_car" route="E_W" depart="723.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_327" type="standard_car" route="N_S" depart="724.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_328" type="standard_car" route="S_E" depart="725.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_329" type="standard_car" route="E_N" depart="727.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_330" type="standard_car" route="N_S" depart="728.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_331" type="standard_car" route="S_E" depart="729.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_332" type="standard_car" route="E_W" depart="731.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_333" type="standard_car" route="S_E" depart="732.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_334" type="standard_car" route="S_N" depart="733.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_335" type="standard_car" route="N_W" depart="736.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_336" type="standard_car" route="S_E" depart="736.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_337" type="standard_car" route="S_N" depart="737.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_338" type="standard_car" route="N_S" depart="737.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_339" type="standard_car" route="E_W" depart="738.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_340" type="standard_car" route="N_E" depart="738.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_341" type="standard_car" route="N_E" depart="739.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_342" type="standard_car" route="W_E" depart="740.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_343" type="standard_car" route="S_E" depart="741.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_344" type="standard_car" route="S_N" depart="742.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_345" type="standard_car" route="W_E" depart="742.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_346" type="standard_car" route="E_W" depart="758.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_347" type="standard_car" route="S_W" depart="759.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_348" type="standard_car" route="N_E" depart="759.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_349" type="standard_car" route="N_S" depart="760.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_350" type="standard_car" route="E_W" depart="760.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_351" type="standard_car" route="E_S" depart="761.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_352" type="standard_car" route="E_W" depart="770.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_353" type="standard_car" route="N_S" depart="772.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_354" type="standard_car" route="S_N" depart="772.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_355" type="standard_car" route="W_E" depart="773.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_356" type="standard_car" route="S_N" depart="782.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_357" type="standard_car" route="E_W" depart="782.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_358" type="standard_car" route="S_N" depart="785.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_359" type="standard_car" route="E_S" depart="800.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_360" type="standard_car" route="N_S" depart="804.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_361" type="standard_car" route="S_N" depart="805.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_362" type="standard_car" route="S_N" depart="808.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_363" type="standard_car" route="W_E" depart="813.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_364" type="standard_car" route="W_E" depart="821.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_365" type="standard_car" route="N_S" depart="822.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_366" type="standard_car" route="E_W" depart="823.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_367" type="standard_car" route="S_E" depart="825.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_368" type="standard_car" route="S_E" depart="828.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_369" type="standard_car" route="N_E" depart="831.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_370" type="standard_car" route="E_W" depart="837.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_371" type="standard_car" route="E_S" depart="838.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_372" type="standard_car" route="S_W" depart="839.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_373" type="standard_car" route="S_N" depart="841.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_374" type="standard_car" route="S_N" depart="843.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_375" type="standard_car" route="S_N" depart="844.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_376" type="standard_car" route="S_N" depart="849.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_377" type="standard_car" route="N_S" depart="849.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_378" type="standard_car" route="W_E" depart="851.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_379" type="standard_car" route="E_W" depart="854.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_380" type="standard_car" route="S_N" depart="855.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_381" type="standard_car" route="E_N" depart="856.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_382" type="standard_car" route="S_W" depart="856.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_383" type="standard_car" route="S_E" depart="859.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_384" type="standard_car" route="N_S" depart="860.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_385" type="standard_car" route="S_N" depart="863.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_386" type="standard_car" route="N_S" depart="866.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_387" type="standard_car" route="S_E" depart="872.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_388" type="standard_car" route="N_S" depart="872.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_389" type="standard_car" route="N_S" depart="879.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_390" type="standard_car" route="E_W" depart="880.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_391" type="standard_car" route="E_S" depart="885.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_392" type="standard_car" route="N_E" depart="886.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_393" type="standard_car" route="E_W" depart="888.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_394" type="standard_car" route="S_N" depart="893.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_395" type="standard_car" route="N_S" depart="893.0" departLane="random" departSpeed="10" />
            <vehicle id="E_N_396" type="standard_car" route="E_N" depart="894.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_397" type="standard_car" route="S_N" depart="911.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_398" type="standard_car" route="E_S" depart="933.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_399" type="standard_car" route="S_N" depart="936.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_400" type="standard_car" route="E_W" depart="963.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_401" type="standard_car" route="E_W" depart="972.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_402" type="standard_car" route="S_N" depart="977.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_403" type="standard_car" route="N_E" depart="980.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_404" type="standard_car" route="E_W" depart="991.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_405" type="standard_car" route="W_E" depart="1000.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_406" type="standard_car" route="E_W" depart="1003.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_407" type="standard_car" route="N_S" depart="1013.0" departLane="random" departSpeed="10" />
            <vehicle id="N_W_408" type="standard_car" route="N_W" depart="1014.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_409" type="standard_car" route="N_S" depart="1022.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_410" type="standard_car" route="N_S" depart="1031.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_411" type="standard_car" route="W_E" depart="1044.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_412" type="standard_car" route="W_S" depart="1063.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_413" type="standard_car" route="W_S" depart="1077.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_414" type="standard_car" route="N_S" depart="1084.0" departLane="random" departSpeed="10" />
            <vehicle id="N_E_415" type="standard_car" route="N_E" depart="1093.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_416" type="standard_car" route="W_E" depart="1105.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_417" type="standard_car" route="W_E" depart="1118.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_418" type="standard_car" route="S_N" depart="1134.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_419" type="standard_car" route="S_N" depart="1135.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_420" type="standard_car" route="E_W" depart="1142.0" departLane="random" departSpeed="10" />
            <vehicle id="E_S_421" type="standard_car" route="E_S" depart="1150.0" departLane="random" departSpeed="10" />
            <vehicle id="S_N_422" type="standard_car" route="S_N" depart="1158.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_423" type="standard_car" route="W_E" depart="1158.0" departLane="random" departSpeed="10" />
            <vehicle id="E_W_424" type="standard_car" route="E_W" depart="1194.0" departLane="random" departSpeed="10" />
            <vehicle id="S_W_425" type="standard_car" route="S_W" depart="1194.0" departLane="random" departSpeed="10" />
            <vehicle id="S_E_426" type="standard_car" route="S_E" depart="1209.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_427" type="standard_car" route="N_S" depart="1223.0" departLane="random" departSpeed="10" />
            <vehicle id="N_S_428" type="standard_car" route="N_S" depart="1243.0" departLane="random" departSpeed="10" />
            <vehicle id="W_S_429" type="standard_car" route="W_S" depart="1436.0" departLane="random" departSpeed="10" />
            <vehicle id="W_E_430" type="standard_car" route="W_E" depart="1556.0" departLane="random" departSpeed="10" />
        </routes>""", file=routes)


        traci.start(self._sumoCmd)

        # set the epsilon for this episode
        self._eps = 1.0 - (episode / self._total_episodes)

        # inits
        self._steps = 0
        tot_neg_reward = 0
        old_total_wait = 0
        self._waiting_times = {}
        self._sum_intersection_queue = 0

        while self._steps < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._steps != 0:
                self._memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._steps != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            if reward < 0:
                tot_neg_reward += reward

        self._save_stats(tot_neg_reward)
        if not self._demo:
            print("Total reward: {}, Eps: {}".format(tot_neg_reward, self._eps))
        traci.close()



    def run_modelless(self):
        action=0
        traci.start(self._sumoCmd)
        self._steps = 0
        tot_neg_reward = 0
        old_total_wait = 0
        self._waiting_times = {}
        self._sum_intersection_queue = 0
        self._eps=0
        while self._steps < self._max_steps:
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait
            self._set_green_phase(action)
            self._simulate(self._green_duration)
            self._set_yellow_phase(action)
            self._simulate(self._yellow_duration)
            old_total_wait = current_total_wait
            if reward < 0:
                tot_neg_reward += reward
            action+=1
            if action==4:
                action=0
        self._save_stats(tot_neg_reward)
        traci.close()

    # HANDLE THE CORRECT NUMBER OF STEPS TO SIMULATE
    def _simulate(self, steps_todo):
        if (self._steps + steps_todo) >= self._max_steps:  # do not do more steps than the maximum number of steps
            steps_todo = self._max_steps - self._steps
        self._steps = self._steps + steps_todo  # update the step counter
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._replay()  # training
            steps_todo -= 1
            intersection_queue = self._get_stats()
            self._sum_intersection_queue += intersection_queue

    # RETRIEVE THE WAITING TIME OF EVERY CAR IN THE INCOMING LANES
    def _get_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[veh_id] = wait_time_car
            else:
                if veh_id in self._waiting_times:
                    del self._waiting_times[veh_id]  # the car isnt in incoming roads anymore, delete his waiting time
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    # DECIDE WHETER TO PERFORM AN EXPLORATIVE OR EXPLOITATIVE ACTION = EPSILON-GREEDY POLICY
    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1) # random action
        else:
            return np.argmax(self._model.predict_one(state, self._sess)) # the best action given the current state

    # SET IN SUMO THE CORRECT YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase("TL", yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    # RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
    def _get_stats(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        intersection_queue = halt_N + halt_S + halt_E + halt_W
        return intersection_queue

    # RETRIEVE THE STATE OF THE INTERSECTION FROM SUMO
    def _get_state(self):
        state = np.zeros(self._model.num_states)

        for veh_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to TL, lane_pos = 0
            lane_group = -1  # just dummy initialization
            valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            # distance in meters from the TLS -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located - _3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7

            if lane_group >= 1 and lane_group <= 7:
                veh_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                veh_position = lane_cell
                valid_car = True

            if valid_car:
                state[veh_position] = 1  # write the position of the car veh_id in the state array

        return state

    # RETRIEVE A GROUP OF SAMPLES AND UPDATE THE Q-LEARNING EQUATION, THEN TRAIN
    def _replay(self):
        batch = self._memory.get_samples(self._model.batch_size)
        if len(batch) > 0:  # if there is at least 1 sample in the batch
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._model.predict_batch(states, self._sess)  # predict Q(state), for every sample
            q_s_a_d = self._model.predict_batch(next_states, self._sess)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._model.num_states))
            y = np.zeros((len(batch), self._model.num_actions))

            for i, b in enumerate(batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._model.train_batch(self._sess, x, y)  # train the NN

    # SAVE THE STATS OF THE EPISODE TO PLOT THE GRAPHS AT THE END OF THE SESSION
    def _save_stats(self, tot_neg_reward):
            self._reward_store.append(tot_neg_reward)  # how much negative reward in this episode
            self._cumulative_wait_store.append(self._sum_intersection_queue)  # total number of seconds waited by cars in this episode
            self._avg_intersection_queue_store.append(self._sum_intersection_queue / self._max_steps)  # average number of queued cars per step, in this episode

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_intersection_queue_store(self):
        return self._avg_intersection_queue_store
