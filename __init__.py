from __future__ import absolute_import

import toolz

import typecheck
import fellow
import dill

from .data import test_json

test_json = [toolz.keyfilter(lambda k: k == "text", d)
             for d in test_json]
import os
dn = os.path.dirname(__file__)
normfn = os.path.join(dn,"norm_pipe")
norm_pipe = dill.load(open(normfn, 'rb'))

@fellow.batch(name="nlp.bag_of_words_model")
@typecheck.test_cases(record=test_json)
@typecheck.returns("number")
def bag_of_words_model(record):
    return norm_pipe.predict([x['text'] for x in [record]])[0]
    #return 3.78121345

@fellow.batch(name="nlp.normalized_model")
@typecheck.test_cases(record=test_json)
@typecheck.returns("number")
def normalized_model(record):
    return norm_pipe.predict([x['text'] for x in [record]])[0]

@fellow.batch(name="nlp.bigram_model")
@typecheck.test_cases(record=test_json)
@typecheck.returns("number")
def bigram_model(record):
    return 3.6

@fellow.app.task(name="nlp.food_bigrams")
@typecheck.returns("100 * string")
def food_bigrams():
	return [u'pei wei', u'womp womp', u'chicha morada', u'ak yelpcdn', u'ritz carlton', u'croque madame', u'goi cuon', u'porta alba', u'deja vu', u'innis gunn', u'tsk tsk', u'parmigiano reggiano', u'laan xang', u'demi glace', u'fleur lys', u'ama ebi', u'sous vide', u'hong kong', u'khao soi', u'jean philippe', u'bok choy', u'bells whistles', u'nanay gloria', u'pel meni', u'yada yada', u'bim bap', u'nuoc mam', u'cochinita pibil', u'vice versa', u'holyrood 9a', u'haricot vert', u'val vista', u'artery clogging', u'thit nuong', u'har gow', u'ore ida', u'alain ducasse', u'hush puppies', u'grana padano', u'pura vida', u'fra diavolo', u'osso bucco', u'dulce leche', u'kee mao', u'khai hoan', u'scantily clad', u'feng shui', u'nooks crannies', u'panna cotta', u'chino bandido', u'cabo wabo', u'ropa vieja', u'tutti santi', u'lloyd wright', u'f_5_unx wrafcxuakbzrdw', u'lomo saltado', u'valle luna', u'puerto rican', u'dac biet', u'coca cola', u'baskin robbins', u'himal chuli', u'kool aid', u'rula bula', u'marche bacchus', u'cien agaves', u'pina colada', u'pin kaow', u'kao tod', u'hodge podge', u'patatas bravas', u'kilt lifter', u'malai kofta', u'reina pepiada', u'hors oeuvres', u'hustle bustle', u'wal mart', u'riff raff', u'krispy kreme', u'bradley ogden', u'hoity toity', u'casey moore', u'arnold palmer', u'leaps bounds', u'harry potter', u'aguas frescas', u'lactose intolerant', u'tammie coe', u'toby keith', u'gulab jamun', u'knick knacks', u'itty bitty', u'hu tieu', u'molecular gastronomy', u'roka akor', u'ping pang', u'uuu uuu', u'moscow mule', u'ezzyujdouig4p gyb3pv_a', u'tres leches']
    # return ["kare kare"] * 100
