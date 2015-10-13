from django import forms


class OptionsForm(forms.Form):
    def __init__(self, *args, **kwargs):
        self.measure_selection = kwargs.pop('selected_measure') if 'selected_measure' in kwargs else 'all'
        self.only_last_iteration = kwargs.pop('only_last_iteration') if 'only_last_iteration' in kwargs else False
        self.measures = self.make_measures_list(kwargs.pop('measures')) if 'measures' in kwargs else\
                        (
                            ('all', 'all'),
                            ('flow_epe', 'flow_epe'),
                            ('epe', 'epe'),
                            ('disp_epe', 'disp_epe'),
                            ('disp_change_epe', 'disp_change_epe'),
                            ('disp_change_err', 'disp_change_err')
                        )

        super(OptionsForm, self).__init__(*args, **kwargs)


        self.fields['selected_measure'] = forms.ChoiceField(label='Error Measure', choices=self.measures, initial=self.measure_selection)
        self.fields['only_last_iteration'] = forms.BooleanField(label='Last Iteration', required=False, initial=self.only_last_iteration)

    def make_measures_list(self, measures):
        result = ('all', 'all'),
        for m in measures:
            result += (m[0], m[0]),
        return result