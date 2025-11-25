import pyvisa as visa
import time
import numpy as np
from matplotlib import pyplot as plt

# Habilito la conexión
rm = visa.ResourceManager()
instrumentos = rm.list_resources()
print(instrumentos)
mux = rm.open_resource(instrumentos[1])
mux.write('*CLS')
print(mux.query('*IDN?'))


# Configuración de par
# input("Introduzca el tiempo entre scans en segundos: ")# Delay (in secs) between scans
ScanInterval = 1
# float(input("Introduzca número de scans a medir: "))#Number of scan sweeps to measure
numberScans = 5
channelDelay = 0.2  # Delay (in secs) between relay closure and measurement
canales = (101, 102)  # List of channels to scan in each scan
ncanales = len(canales)

# Format configuration
# set the channel list to scan
mux.write('ROUTE:SCAN', '(@' + str(canales)[1:])
mux.write('ROUT:CHAN:DELAY ' + str(channelDelay))
# ? Return channel number with each reading
mux.write('FORMAT:READING:CHAN ON')
mux.write('FORMAT:READING:TIME ON')  # ? Return time stamp with each reading
# elige el formato para el tiempo
mux.write('FORMat:READing:TIME:TYPE  ABSolute')
mux.write('FORMat:READing:UNIT OFF')
mux.write('TRIG:TIMER ' + str(ScanInterval))  # Delay (in secs) between scans

a = mux.query_ascii_values('SYSTEM:TIME?')  # pido la hora de inicio
print(a)
t0 = float(a[0])*3600+float(a[1])*60+a[2]
print(t0)
datos = []

plt.close("all")

i = 0
for i in range(numberScans):
    #        NumeroDeMedicion = i #para saber por que medicion vamos
    # start scan
    data = mux.query_ascii_values('READ?')
#        mux.write('INIT')
#        data = mux.query_ascii_values('FETC?')

    # wait to the end of the scan
    time.sleep(.5 + channelDelay*ncanales)

#        print(data)
    i += 1

    # temp = float(mux.query('MEASURE:TEMP?'))
    print(i, data)

    plt.scatter(i, data[0], color="b")  # ,'ob', ms=3)
    plt.scatter(i, data[8], color="c")
    plt.pause(0.3)


mux.close()

# difusividad=np.array(data)
# difusividad.to_csv('Mediciones'+str(time.localtime()[0])+'-'+str(time.localtime()[1])+'-'+str(time.localtime()[2])+'-'+str(time.localtime()[3])+'-'+str(time.localtime()[4])+'-'+str(time.localtime()[5])+'.csv')


# Otras cosas que se pueden medir
#
# MEASure
#     :VOLTage:DC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :VOLTage:DC:RATio? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :VOLTage:AC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :CURRent:DC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :CURRent:AC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :RESistance? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :FRESistance? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :FREQuency? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :PERiod? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#     :CONTinuity?
#     :DIODe?
