######################################################№########
#           PublicApi-клиент TraderNet для Python 3. V 0.1.3  #
#           author:  Mr. TradeBot & Robotrader Jr             #
#           mail:    mr.tradebot@gmail.com                    #
#                                                             #
# https://tradernet.kz/r/PublicApiClient_v0.1.3.py.zip        #
#                                                             #
#######################################################№#######

import time, hmac, hashlib, requests, json, urllib3


class PublicApiClient:
    """
    Имена приватных переменных класса должны начинаться на два подчеркивания: __
    """
    V1: int = 1
    V2: int = 2
    __apiUrl = str()
    __apiKey = str()
    __apiSecret = str()
    __version = int()
    __devMode: bool = False

    def __init__(self, apiKey=None, apiSecret=None, 
                 version=V1):
        """
        Инициализация экземпляра класса
        :param apiKey:
        :param apiSecret:
        :param version:
        """
        self.__apiUrl = 'https://tradernet.global/api' 
        self.__version = version
        self.__apiKey = apiKey
        self.__apiSecret = apiSecret

    def setApiUrl(self, _apiUrl):
        """
        подгружаем нужный URL, если не устраивает дефолтный
        :param _apiUrl:
        :return:
        """
        self.__apiUrl = _apiUrl

    def isDevMode(self):
        """
        Включаем режим разработки
        :return:
        """
        self.__devMode = True

    def preSign(self, d):
        """
        preSign используется для подписи с ключом
        :param d:
        :return: string
        """

        s = ''
        for i in sorted(d):
            if type(d[i]) == dict:
                s += i + '=' + self.preSign(d[i]) + '&'
            else:
                s += i + '=' + str(d[i]) + '&'
        return s[:-1]

    def httpencode(self, d):
        """
        httpencode - аналог функции http_build_query для URL-запроса, обновленный , с работой с вложенными списками
        :param d:
        :return: string
        """
        s = ''
        for i in sorted(d):
            if type(d[i]) == dict:
                for into in d[i]:
                    if type(d[i][into]) == dict:
                        for subInto in d[i][into]:
                            if type(d[i][into][subInto]) == dict:
                                s += self.httpencode(d[i][into][subInto])
                            else:
                                s += i + '[' + into + ']' + '[' + subInto + ']=' + str(d[i][into][subInto]) + '&'
                    else:
                        s += i + '[' + into + ']=' + str(d[i][into]) + '&'
            else:
                s += i + '=' + str(d[i]) + '&'

        return s[:-1]

    def sendRequest(self, method, aParams=None, format='JSON'):
        """
        Отправка запроса
        :param method:
        :param aParams:
        :param format:
        :return: Responce
        """

        aReq = dict()
        aReq['cmd'] = method
        if aParams:
            aReq['params'] = aParams
        if (self.__version != self.V1) and (self.__apiKey):
            aReq['apiKey'] = self.__apiKey
        aReq['nonce'] = int(time.time() * 10000)

        preSig = self.preSign(aReq)
        Presig_Enc = self.httpencode(aReq)

        # Игнорим ошибки для локального соединения по ssl
        isVerify = True

        if self.__devMode == True:
            urllib3.disable_warnings()
            isVerify = False

        # Создание подписи и выполнение запроса в зависимости от V1 или V2
        if self.__version == self.V1:
            aReq['sig'] = hmac.new(key=self.__apiSecret.encode(), digestmod=hashlib.sha256).hexdigest()
            res = requests.post(self.__apiUrl, data={'q': json.dumps(aReq)}, verify=isVerify)
        else:
            apiheaders = {
                'X-NtApi-Sig': hmac.new(key=self.__apiSecret.encode(), msg=preSig.encode('utf-8'),
                                        digestmod=hashlib.sha256).hexdigest(),
                'Content-Type': 'application/x-www-form-urlencoded'
                # Нужно в явном виде указать Content-Type, иначе не будет работать;
                # по какой-то причине requests.post не может сам это сделать
            }
            self.__apiUrl += '/v2/cmd/' + method
            res = requests.post(self.__apiUrl, params=Presig_Enc, headers=apiheaders, data=Presig_Enc, verify=isVerify)

        return (res)