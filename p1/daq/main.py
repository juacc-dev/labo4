import pylabo.plot as plot
from core import list_devices, measure


print(list_devices())


channel_name = "Dev1/ai1"
duration = 1
freq = 250000

df = measure(
    channel_name,
    duration,
    freq
)

df.to_csv("test.csv")

plot.data(df)
plot.show()
