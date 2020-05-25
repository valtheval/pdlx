from modelling import rule_base_classifier
import numpy as np


def test_set_params():
    model = rule_base_classifier.RuleBaseClassifier()
    params = {"rule_size":15, "context_size":10, "exclude_words":["oui"]}
    model.set_params(params)
    assert model.rule_size == 15
    assert model.context_size == 10
    assert model.exclude_words[0] == "oui"
    assert len(model.exclude_words) == 1


def test_fit():
    X = np.array([['monsieur', 'x', 'a', 'victime', 'accident', 'travail', '19910409', 'cours', 'duquel', 'a',
                  'gravement', 'blessé', "20200424", 'suivant', 'jugement', 'date', 'confirmé', 'arrêt', 'cour', 'date', '19990330',
                  'tribunal', 'affaires', 'sécurité', 'sociale', 'lot', 'garonne', 'a', 'dit', 'cet', 'accident'],

                  ['bias', 'repassistant', 'pascale', 'luguet', '20200424', 'avocat', 'barreau', 'agen', 'appelants',
                   'jugement', 'tribunal', 'affaires', 'sécurité', 'sociale', 'agen', 'date', 'part', 'groupe', 'azur',
                   'assurances', 'iard', '7', "19970424", 'avenue', 'marcel', 'proust', '28032', 'chartres', 'cedex',
                   'repassistant', 'jeanloup', 'bourdin'],

                  ['accident', 'travail', '19970424', 'cours', 'duquel', 'a', 'gravement', 'blessé', 'suivant',
                   'jugement', 'date', '19910409', 'confirmé', 'arrêt', 'cour', 'date', 'tribunal', 'affaires', 'sécurité',
                   'sociale', 'lot', 'garonne', 'a', 'dit', 'cet', 'accident', 'dû', 'faute', 'inexcusable', 'sarl',
                   'transports']])
    y = np.array(["19910409", "20200424", "19970424"])
    model = rule_base_classifier.RuleBaseClassifier(rule_size=5)
    model.fit(X, y)
    assert all(a == b for a, b in zip(model.rule, ['a', 'accident', 'travail', 'cours', 'duquel']))

    model = rule_base_classifier.RuleBaseClassifier(rule_size=5, exclude_words=["a"])
    model.fit(X, y)
    assert all(a == b for a, b in zip(model.rule, ['accident', 'travail', 'cours', 'duquel', "gravement"]))


def test_predict():
    X = np.array([['monsieur', 'x', 'a', 'victime', 'accident', 'travail', '19910409', 'cours', 'duquel', 'a',
                   'gravement', 'blessé', "20200424", 'suivant', 'jugement', 'date', 'confirmé', 'arrêt', 'cour',
                   'date', '19990330', 'tribunal', 'affaires', 'sécurité', 'sociale', 'lot', 'garonne', 'a', 'dit',
                   'cet', 'accident'],

                  ['bias', 'repassistant', 'pascale', 'luguet', '20200424', 'avocat', 'barreau', 'agen', 'appelants',
                   'jugement', 'tribunal', 'affaires', 'sécurité', 'sociale', 'agen', 'date', 'part', 'groupe', 'azur',
                   'assurances', 'iard', 'accident', "19970424", 'avenue', 'marcel', 'proust', '28032', 'chartres', 'cedex',
                   'repassistant', 'jeanloup', 'bourdin'],

                  ["test", "test3", "test2", 'accident', 'travail', '19880525', 'cours', 'duquel', 'a', 'gravement', 'blessé', 'suivant',
                   'jugement', 'date', '19910409', 'confirmé', 'arrêt', 'cour', 'date', 'tribunal', 'affaires',
                   'sécurité', 'sociale', 'lot', 'garonne', 'a', 'dit', 'cet', 'accident', 'dû', 'faute', 'inexcusable',
                   'sarl', 'transports']])

    y = np.array(["19910409", "20200424", "19970424"])
    model = rule_base_classifier.RuleBaseClassifier(rule_size=5)
    model.fit(X, y)
    # ici model.rule = ['a', 'x', 'victime', 'accident', 'travail']
    y_pred = model.predict(X)
    assert all(a == b for a, b in zip(y_pred, ["19910409", "19970424", "19880525"]))
    print(model.proba)
    assert all(a == b for a,b in zip(model.proba, [0.5, 0.1, 0.3]))
