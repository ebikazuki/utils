from discordwebhook import Discord

discord = Discord(url="https://discordapp.com/api/webhooks/1270279845536071720/utDJl0NIcQVBCbssjEKX3jrPMMe3d4NnmVnrdEi4_l2Xg1qiysQrHKv18_LOTx-pzirD")
discord.post(content="Hello, world.")