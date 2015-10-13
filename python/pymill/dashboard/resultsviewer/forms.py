from django import forms


class MeasureSelectorForm(forms.Form):
    def __init__(self, *args, **kwargs):
        if 'selected_measure' in kwargs:
            self.selection = kwargs.pop('selected_measure')
        else:
            self.selection = 'flow_epe'
        super(MeasureSelectorForm, self).__init__(*args, **kwargs)

        self.measures = (
                    ('all', 'all'),
                    ('flow_epe', 'flow_epe'),
                    ('disp_epe', 'disp_epe'),
                    ('disp_change_epe', 'disp_change_epe'),
                    ('disp_change_err', 'disp_change_err')
                    )
        self.fields['selected_measure'] = forms.ChoiceField(label='Error Measure', choices=self.measures, initial=self.selection)
