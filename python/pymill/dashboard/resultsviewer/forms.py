from django import forms


class OptionsForm(forms.Form):
    def __init__(self, *args, **kwargs):
        self.measure_selection = kwargs.pop('selected_measure') if 'selected_measure' in kwargs else 'all'
        self.only_last_iteration = kwargs.pop('only_last_iteration') if 'only_last_iteration' in kwargs else False


        super(OptionsForm, self).__init__(*args, **kwargs)

        self.measures = (
                    ('all', 'all'),
                    ('flow_epe', 'flow_epe'),
                    ('epe', 'epe'),
                    ('disp_epe', 'disp_epe'),
                    ('disp_change_epe', 'disp_change_epe'),
                    ('disp_change_err', 'disp_change_err')
                    )
        self.fields['selected_measure'] = forms.ChoiceField(label='Error Measure', choices=self.measures, initial=self.measure_selection)
        self.fields['only_last_iteration'] = forms.BooleanField(label='Last Iteration', required=False, initial=self.only_last_iteration)