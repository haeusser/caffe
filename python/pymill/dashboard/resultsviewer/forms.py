from django import forms


class OptionsForm(forms.Form):
    def __init__(self, *args, **kwargs):
        self.only_last_iteration = kwargs.pop('only_last_iteration') if 'only_last_iteration' in kwargs else False
        self.measure_selection = kwargs.pop('selected_measure') if 'selected_measure' in kwargs else ''
        self.measures = self.make_dropdown_list(kwargs.pop('measures')) if 'measures' in kwargs else\
                        (
                            ('', 'all'),
                            ('flow_epe', 'flow_epe'),
                            ('epe', 'epe'),
                            ('disp_epe', 'disp_epe'),
                            ('disp_change_epe', 'disp_change_epe'),
                            ('disp_change_err', 'disp_change_err')
                        )
        self.position_selection = kwargs.pop('selected_position') if 'selected_position' in kwargs else ''
        self.positions =self.make_dropdown_list(kwargs.pop('positions')) if 'positions' in kwargs else\
                        (
                            ('', 'all'),
                            ('L', 'L'),
                            ('R', 'R'),
                            ('0L', '0L'),
                            ('0R', '0R'),
                        )

        super(OptionsForm, self).__init__(*args, **kwargs)

        self.fields['selected_measure'] = forms.ChoiceField(label='Error Measure', choices=self.measures, initial=self.measure_selection, required=False)
        self.fields['selected_position'] = forms.ChoiceField(label='Position', choices=self.positions, initial=self.position_selection, required=False)
        self.fields['only_last_iteration'] = forms.BooleanField(label='Last Iteration', required=False, initial=self.only_last_iteration)

    def make_dropdown_list(self, measures):
        result = ('', 'all'),
        for m in measures:
            if m not in [[''], [None]]:
                result += (m[0], m[0]),
        return result

class RestartForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(RestartForm, self).__init__(*args, **kwargs)
        self.fields['passwd'] = forms.CharField(widget=forms.PasswordInput(), label='', required=False)


