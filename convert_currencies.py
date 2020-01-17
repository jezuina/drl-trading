instruments = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',  'GBP/USD', 'NZD/USD', 'GBP/JPY', 'EUR/JPY', 'AUD/JPY', 'EUR/GBP', 'USD/CHF']
current_prc = [ 0.8452,    108.66250, 0.70304,   1.29570,   1.5235,    1.2125,    144.25800,     125.43,    76.40250,  0.84592,   0.9547]
lot_size = 1

#  https://www.xm.com/forex-calculators/pip-value

def calculate_pip_value_in_account_currency(currency, current_prices):
    pip_values = []
    currencysplitted = currency.split('/')
    first_currency = currencysplitted[0]
    second_currency = currencysplitted[1]

    current_price_currency = 1
    value_convert_to_usd = 1

    for x in range(len(current_prices)):
        if instruments[x] == currency:
            current_price_currency = current_prices[x]
            lotsizenow = lot_size

    for x in range(len(instruments)):
        if second_currency != 'USD':
            convert_currency = first_currency + '/USD'
            if instruments[x] == convert_currency:
                value_convert_to_usd = current_prices[x]


        #pip value multiplied by the bid/ask currency pair times the lot size = the price per pip
    if second_currency != 'JPY':    #if there is no JPY in the second currency the pip value is always 0.01% = 0.0001 of the current bid/ask
        pip_value = 0.0001 / current_price_currency * lotsizenow * 100000 * value_convert_to_usd
    elif second_currency == 'JPY':
        pip_value = 0.01 / current_price_currency * lotsizenow * 100000 * value_convert_to_usd #if the second currency is JPY the pip value is 1% = 0.01 of the current bid/ask

    pip_values.append(pip_value) #adds the pip value to an array and returns it

    return pip_values

def _calculate_pip_value_in_account_currency(currency, current_prices):
        pip_values = []
        #   print(type(current_prices))
        #   if currency == account_currency.USD:
        if currency == 'USD':
            m = 0
            #   print(self.instruments)
            for instrument in instruments:
                #   print(instrument)
                if instrument == 'EUR/USD':
                    EUR_USD = current_prices[m]
                    USD_EUR = 1/current_prices[m]
                elif instrument == 'USD/JPY':
                    USD_JPY = current_prices[m]
                    JPY_USD = 1/current_prices[m]
                elif instrument == 'AUD/USD':
                    AUD_USD = current_prices[m]
                    USD_AUD = 1/current_prices[m]
                elif instrument == 'GBP/USD':
                    GBP_USD = current_prices[m]
                    USD_GBP = 1/current_prices[m]
                
                currency = instrument.split('/')
                first_currency = currency[0]
                second_currency = currency[1]
                
                if first_currency == 'USD' and second_currency != 'JPY':
                    pip_value = 10/current_prices[m]
                elif second_currency == 'USD':
                    pip_value = 10
                elif first_currency == 'USD' and second_currency == 'JPY':
                    pip_value = 0.01/current_prices[m]    
                elif instrument == 'GBP/JPY':
                    pip_value = GBP_USD/current_prices[m] 
                elif instrument == 'EUR/JPY':
                    pip_value = EUR_USD * 0.01/current_prices[m] 
                elif instrument == 'AUD/JPY':
                    pip_value = AUD_USD * 0.01/current_prices[m] 
                elif instrument == 'EUR/GBP':
                    pip_value = current_prices[m]/USD_GBP 

                pip_values.append(pip_value)
                m += 1    

        return pip_values 
    
if __name__ == '__main__':
    print(instruments)
    print(_calculate_pip_value_in_account_currency('USD', current_prc))